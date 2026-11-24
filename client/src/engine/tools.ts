import { nanoid } from 'nanoid';
import { eraseStroke } from '../../../shared/erase';
import {
  frameOf, highlighterIndex, hitObject, matMul, matRotateAbout, matScaleAbout, matTranslate, objBounds, objInLasso, objInRect,
  pointInFrame, segHitsInk, shapeToStrokePoints, strokeBBox, transformObject, type Frame, type Mat,
} from '../../../shared/geometry';
import { round1, round2, straightPoints } from '../../../shared/ink';
import { isScribble, recognizeShape, type Recognized } from '../../../shared/recognize';
import { findSnap, knobPos, rulerContains, snapAngle, snapPoint, type Snap } from '../../../shared/ruler';
import { dragCrop, toImageFraction } from '../../../shared/crop';
import { spaceTargets } from '../../../shared/space';
import { MultiTapDetector } from '../../../shared/touchTap';
import { CLIP_PREFIX, type Axes, type BBox, type BoardObject, type Crop, type Graph, type ImageObj, type Shape, type Stroke, type StrokeStyle, type Tape } from '../../../shared/types';
import type { Engine, OpBody } from './Engine';
import { drawLiveStroke, drawShape, gridUnit, resolveColor } from './render';
import { ERASER_SIZES, type ShapeTool } from './store';

export interface Pt { x: number; y: number; p: number; sx: number; sy: number }

interface Gesture {
  move(pts: Pt[], ev: PointerEvent): void;
  up(ev: PointerEvent): void;
  cancel(): void;
  draw?(ctx: CanvasRenderingContext2D): void;
  frame?(now: number): void;
  kind: string;
}

const GRAPH_COLORS = ['#1971c2', '#e03131', '#2f9e44', '#f08c00', '#9c36b5', '#0c8599'];
export const nextGraphColor = (n: number) => GRAPH_COLORS[n % GRAPH_COLORS.length];

const isEditable = (t: EventTarget | null) => {
  const el = t as HTMLElement | null;
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);
};
/** Sliders, checkboxes and colour pickers: they take arrow keys, but not undo / redo or Esc. */
const isControl = (t: EventTarget | null) => {
  const el = t as HTMLInputElement | null;
  return !!el && el.tagName === 'INPUT' && ['range', 'checkbox', 'radio', 'color', 'button'].includes(el.type);
};

function snapTo15(ax: number, ay: number, bx: number, by: number): [number, number] {
  const d = Math.hypot(bx - ax, by - ay);
  const a = Math.round(Math.atan2(by - ay, bx - ax) / (Math.PI / 12)) * (Math.PI / 12);
  return [ax + d * Math.cos(a), ay + d * Math.sin(a)];
}

/** A light ring with a dark ring inside it: one of the two always shows, on white paper, dark photos or a dark board. */
function haloRing(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, alpha: number) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 1.25;
  ctx.stroke();
  ctx.restore();
}

// =====================================================================
// Pen / highlighter
// =====================================================================
class PenGesture implements Gesture {
  kind = 'pen';
  id = nanoid(12);
  pts: number[] = [];
  private unsent: number[] = [];
  private reset = false;
  private style: StrokeStyle;
  private mode: 'free' | 'edge' | 'line';
  private lineStart: [number, number];
  private hold: { rec: Recognized; base: Shape; at: [number, number] } | null = null;
  private holdShape: Shape | null = null;
  private anchor: { sx: number; sy: number; t: number };
  private lastS: [number, number];
  private travel = 0;

  constructor(private e: Engine, tool: 'pen' | 'highlighter', p: Pt, ev: PointerEvent, private snap: Snap | null) {
    this.style = e.currentStyle(tool, ev.pointerType !== 'pen');
    const straightHl = this.style.pen === 'highlighter' && (ev.shiftKey || e.settings.hl.straight);
    this.mode = snap ? (snap.type === 'ray' ? 'line' : 'edge') : straightHl ? 'line' : 'free';
    this.lineStart = snap?.type === 'ray' ? [snap.cx, snap.cy] : [round1(p.x), round1(p.y)];
    this.anchor = { sx: p.sx, sy: p.sy, t: performance.now() };
    this.lastS = [p.sx, p.sy];
    e.sync.emit('live:start', { strokeId: this.id, pageId: e.pageId, style: this.style });
    this.add(p, ev);
  }

  private pressure(p: Pt) {
    return this.style.sim ? 0.5 : round2(Math.max(0.05, Math.min(1, p.p || 0.5)));
  }

  private add(p: Pt, ev: PointerEvent) {
    if (this.mode === 'line') {
      let end: [number, number] = this.snap ? snapPoint(this.snap, p.x, p.y) : [p.x, p.y];
      if (!this.snap && ev.shiftKey && this.style.pen !== 'highlighter') end = snapTo15(this.lineStart[0], this.lineStart[1], end[0], end[1]);
      this.pts = straightPoints(this.lineStart[0], this.lineStart[1], end[0], end[1]);
      this.reset = true;
      return;
    }
    const [wx, wy] = this.mode === 'edge' && this.snap ? snapPoint(this.snap, p.x, p.y) : [p.x, p.y];
    const x = round1(wx), y = round1(wy);
    const n = this.pts.length;
    if (n && this.pts[n - 3] === x && this.pts[n - 2] === y) return;
    const pr = this.mode === 'edge' ? 0.5 : this.pressure(p);
    this.pts.push(x, y, pr);
    this.unsent.push(x, y, pr);
  }

  move(pts: Pt[], ev: PointerEvent) {
    const now = performance.now();
    for (const p of pts) {
      if (Math.hypot(p.sx - this.anchor.sx, p.sy - this.anchor.sy) > 4) this.anchor = { sx: p.sx, sy: p.sy, t: now };
      this.travel += Math.hypot(p.sx - this.lastS[0], p.sy - this.lastS[1]);
      this.lastS = [p.sx, p.sy];
      if (this.hold) this.adjust(p, ev); else this.add(p, ev);
    }
    this.e.invalidateLive();
  }

  frame(now: number) {
    // draw-and-hold: pen kept still for ~500 ms at the end of a stroke
    if (!this.hold && this.mode === 'free' && this.travel > 24 && now - this.anchor.t > 500) this.triggerHold();
    if (this.reset) {
      const points = this.holdShape ? shapeToStrokePoints(this.holdShape).flat() : this.pts;
      this.e.sync.emit('live:points', { strokeId: this.id, points, reset: true });
    } else if (this.unsent.length) {
      this.e.sync.emit('live:points', { strokeId: this.id, points: this.unsent });
    }
    this.unsent = [];
    this.reset = false;
  }

  private makeShape(rec: Recognized): Shape {
    return {
      id: this.id, type: 'shape', authorId: this.e.me.userId, createdAt: Date.now(), kind: rec.kind, points: rec.points,
      color: this.style.color, size: this.style.size, ...(rec.rotation ? { rotation: rec.rotation } : {}),
    };
  }

  private triggerHold() {
    const n = this.pts.length;
    if (n < 6) return;
    const last: [number, number] = [this.pts[n - 3], this.pts[n - 2]];
    if (this.style.pen === 'highlighter') {
      // highlighters stay highlighters: straighten into a stroke
      this.mode = 'line';
      this.lineStart = [this.pts[0], this.pts[1]];
      this.pts = straightPoints(this.lineStart[0], this.lineStart[1], last[0], last[1]);
      this.reset = true;
    } else {
      const rec = recognizeShape(this.pts, { scale: this.e.cam.zoom, mode: 'hold' })
        ?? { kind: 'line' as const, points: [this.pts[0], this.pts[1], last[0], last[1]] };
      const base = this.makeShape(rec);
      this.hold = { rec, base, at: last };
      this.holdShape = base;
      this.reset = true;
    }
    this.e.invalidateLive();
  }

  /** While still holding the pen down after a snap: adjust the endpoint (lines) or the size (closed shapes). */
  private adjust(p: Pt, ev: PointerEvent) {
    const { base, at } = this.hold!;
    if (base.kind === 'line' || base.kind === 'arrow') {
      let end: [number, number] = [round1(p.x), round1(p.y)];
      if (ev.shiftKey) end = snapTo15(base.points[0], base.points[1], end[0], end[1]);
      this.holdShape = { ...base, points: [base.points[0], base.points[1], round1(end[0]), round1(end[1])] };
    } else {
      const f = frameOf(base);
      let cx = 0, cy = 0;
      if (f) { cx = f.cx; cy = f.cy; } else {
        const n = base.points.length / 2;
        for (let i = 0; i < n; i++) { cx += base.points[i * 2] / n; cy += base.points[i * 2 + 1] / n; }
      }
      const d0 = Math.hypot(at[0] - cx, at[1] - cy);
      const k = d0 > 2 ? Math.max(0.05, Math.min(50, Math.hypot(p.x - cx, p.y - cy) / d0)) : 1;
      this.holdShape = { ...base, ...(transformObject(base, matScaleAbout(k, k, cx, cy)) as Partial<Shape>), size: base.size } as Shape;
      if (this.holdShape.rotation === null) delete this.holdShape.rotation;
    }
    this.reset = true;
  }

  draw(ctx: CanvasRenderingContext2D) {
    if (this.holdShape) drawShape(ctx, this.holdShape, this.e.env);
    else drawLiveStroke(ctx, this.style, this.pts, this.e.env);
  }

  private stroke(id: string): Stroke {
    return {
      id, type: 'stroke', authorId: this.e.me.userId, createdAt: Date.now(), pen: this.style.pen, color: this.style.color,
      size: this.style.size, opacity: this.style.opacity, points: this.pts, bbox: strokeBBox(this.pts), stab: this.style.stab,
      ...(this.style.sim ? { sim: true } : {}),
    };
  }

  up() {
    const e = this.e;
    this.frame(performance.now());
    e.sync.emit('live:end', { strokeId: this.id });
    if (this.pts.length < 3) return;

    if (this.holdShape) {
      e.commit([{ kind: 'add', obj: this.holdShape }]);
      return;
    }
    const s = e.settings;
    if (this.mode === 'free' && s.scribbleErase && isScribble(this.pts, e.cam.zoom)) {
      const hit = this.scribbleTargets();
      if (hit.length) { e.commit(hit.map((id) => ({ kind: 'delete', id }))); return; }
    }
    if (this.mode === 'free' && s.inkToShape && this.style.pen !== 'highlighter') {
      const rec = recognizeShape(this.pts, { scale: e.cam.zoom, mode: 'auto' });
      if (rec) {
        // The clean shape takes the stroke's id (it replaces the live preview). Undo history gets
        // two steps, so one Ctrl+Z brings back the original ink and a second removes it.
        const ink = this.stroke(nanoid(12));
        const shape = this.makeShape(rec);
        e.commit([{ kind: 'add', obj: shape }], { undo: false });
        e.pushUndo([{ kind: 'delete', id: ink.id }]);
        e.pushUndo([{ kind: 'delete', id: shape.id }, { kind: 'add', obj: ink }]);
        return;
      }
    }
    const stroke = this.stroke(this.id);
    if (stroke.pen === 'highlighter') {
      // highlighter goes beneath the ink it covers
      const b = stroke.bbox, pad = stroke.size / 2;
      const index = highlighterIndex(e.page, [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]);
      e.commit([{ kind: 'add', obj: stroke, ...(index < e.page.order.length ? { index } : {}) }]);
    } else e.commit([{ kind: 'add', obj: stroke }]);
  }

  private scribbleTargets(): string[] {
    const page = this.e.page;
    const b = strokeBBox(this.pts);
    const out: string[] = [];
    for (const id of page.order) {
      const o = page.objects[id];
      if (!o || o.locked || (o.type !== 'stroke' && o.type !== 'shape')) continue;
      const ob = objBounds(o);
      if (ob[0] > b[2] || ob[2] < b[0] || ob[1] > b[3] || ob[3] < b[1]) continue;
      for (let i = 3; i < this.pts.length; i += 3) {
        if (segHitsInk(o, this.pts[i - 3], this.pts[i - 2], this.pts[i], this.pts[i + 1], 1)) { out.push(id); break; }
      }
    }
    return out;
  }

  cancel() {
    this.e.sync.emit('live:end', { strokeId: this.id });
  }
}

// =====================================================================
// Eraser (stroke + partial)
// =====================================================================
class EraserGesture implements Gesture {
  kind = 'eraser';
  private last: Pt;
  private origs = new Map<string, { obj: BoardObject; index: number }>();
  private created = new Set<string>();
  private initialIndex = new Map<string, number>();
  private done = false;

  constructor(private e: Engine, p: Pt) {
    this.last = p;
    e.page.order.forEach((id, i) => this.initialIndex.set(id, i));
    this.sweep(p, p);
  }

  get radius() { return ERASER_SIZES[this.e.settings.eraser.size] / this.e.cam.zoom; }

  private track(o: BoardObject) {
    if (this.created.has(o.id)) this.created.delete(o.id);
    else if (!this.origs.has(o.id)) this.origs.set(o.id, { obj: o, index: this.initialIndex.get(o.id) ?? 1e9 });
  }

  private sweep(a: Pt, b: Pt) {
    const e = this.e, page = e.page, s = e.settings.eraser, r = this.radius;
    const ops: OpBody[] = [];
    let delta = 0;
    for (let i = 0; i < page.order.length; i++) {
      const o = page.objects[page.order[i]];
      if (!o || o.locked || (o.type !== 'stroke' && o.type !== 'shape')) continue; // images, graphs, axes, text are never erasable
      if (s.hlOnly && !(o.type === 'stroke' && o.pen === 'highlighter')) continue;
      if (!segHitsInk(o, a.x, a.y, b.x, b.y, r)) continue;
      if (s.mode === 'stroke') {
        ops.push({ kind: 'delete', id: o.id });
        this.track(o);
        delta -= 1;
        continue;
      }
      let pieces: Stroke[];
      if (o.type === 'stroke') {
        const cut = eraseStroke(o.points, a.x, a.y, b.x, b.y, r + o.size * 0.2);
        if (!cut) continue;
        pieces = cut.map((points) => ({ ...o, id: nanoid(12), points, bbox: strokeBBox(points) }));
      } else {
        // shapes are turned into plain strokes so they can be partially erased
        pieces = [];
        for (const poly of shapeToStrokePoints(o)) {
          const cut = eraseStroke(poly, a.x, a.y, b.x, b.y, r + o.size * 0.2) ?? [poly];
          for (const points of cut) {
            pieces.push({
              id: nanoid(12), type: 'stroke', authorId: o.authorId, createdAt: o.createdAt, pen: 'ballpoint', color: o.color,
              size: o.size, opacity: 1, points, bbox: strokeBBox(points), stab: 0,
            });
          }
        }
      }
      ops.push({ kind: 'delete', id: o.id });
      this.track(o);
      pieces.forEach((obj, j) => {
        ops.push({ kind: 'add', obj, index: i + delta + j });
        this.created.add(obj.id);
      });
      delta += pieces.length - 1;
    }
    if (ops.length) e.commit(ops, { undo: false });
    e.sync.emit('live:erase', { pageId: e.pageId, x: round1(b.x), y: round1(b.y), r: round1(r), active: true });
  }

  move(pts: Pt[]) {
    for (const p of pts) { this.sweep(this.last, p); this.last = p; }
    this.e.invalidateLive();
  }

  private finish() {
    if (this.done) return;
    this.done = true;
    const e = this.e, page = e.page;
    e.sync.emit('live:erase', { pageId: e.pageId, x: 0, y: 0, r: 0, active: false });
    // one undo step for the whole gesture: remove surviving pieces, put the originals back in place
    const undo: OpBody[] = [];
    for (const id of this.created) if (page.objects[id]) undo.push({ kind: 'delete', id });
    for (const { obj, index } of [...this.origs.values()].sort((x, y) => x.index - y.index)) undo.push({ kind: 'add', obj, index });
    if (this.origs.size) e.pushUndo(undo);
  }
  up() { this.finish(); }
  cancel() { this.finish(); }
}

// =====================================================================
// Select: lasso / rectangle / click, move, resize, rotate
// =====================================================================
class SelectGesture implements Gesture {
  kind = 'select';
  private mode: 'lasso' | 'rect' | 'move' | 'scale' | 'rotate';
  private path: number[] = [];
  private cur: Pt;
  private moved = false;
  private ids = new Set<string>();
  private f0: Frame | null = null;
  private handle: { hx: number; hy: number } | null = null;
  private m: Mat = [1, 0, 0, 1, 0, 0];
  private clicked: BoardObject | null = null;
  /** the only selected object when the press began (a tap on a selected graph reads coordinates) */
  private onlySel: string | null;
  private touch: boolean;

  constructor(private e: Engine, private input: Input, private start: Pt, ev: PointerEvent) {
    this.cur = start;
    this.onlySel = e.sel.size === 1 ? [...e.sel][0] : null;
    this.touch = ev.pointerType === 'touch';
    const z = e.cam.zoom;
    const grab = (ev.pointerType === 'touch' ? 18 : 11) / z;
    const f = e.selFrame();
    this.mode = e.settings.select.mode;
    if (f && !e.selLocked()) {
      for (const h of e.handles(f)) {
        if (Math.hypot(start.x - h.x, start.y - h.y) <= grab) {
          this.mode = h.rotate ? 'rotate' : 'scale';
          this.handle = h;
          this.f0 = f;
          break;
        }
      }
    }
    if (this.mode !== 'scale' && this.mode !== 'rotate') {
      if (f && !ev.shiftKey && !e.selLocked() && pointInFrame(f, start.x, start.y, 2 / z)) this.mode = 'move';
      else {
        const hit = this.hitTop(false);
        if (hit) {
          if (ev.shiftKey) {
            const next = new Set(e.sel);
            if (next.has(hit.id)) next.delete(hit.id); else next.add(hit.id);
            e.setSelection(next);
          } else if (!e.sel.has(hit.id)) e.setSelection([hit.id]);
          this.clicked = hit;
          this.mode = 'move';
        }
      }
    }
    if (this.mode === 'move' || this.mode === 'scale' || this.mode === 'rotate') {
      for (const o of e.selObjects()) if (!o.locked) this.ids.add(o.id);
      if (!this.clicked && this.mode === 'move') this.clicked = this.hitTop(false);
    } else this.path.push(start.x, start.y);
  }

  private hitTop(locked: boolean): BoardObject | null {
    const page = this.e.page, r = 7 / this.e.cam.zoom;
    for (let i = page.order.length - 1; i >= 0; i--) {
      const o = page.objects[page.order[i]];
      if (!o || !!o.locked !== locked) continue;
      if (hitObject(o, this.start.x, this.start.y, r)) return o;
    }
    return null;
  }

  move(pts: Pt[], ev: PointerEvent) {
    const p = pts[pts.length - 1];
    this.cur = p;
    const e = this.e;
    if (!this.moved && Math.hypot(p.sx - this.start.sx, p.sy - this.start.sy) < 3) return;
    this.moved = true;
    if (this.mode === 'lasso') {
      for (const q of pts) this.path.push(q.x, q.y);
    } else if (this.mode === 'move') {
      this.m = matTranslate(round1(p.x - this.start.x), round1(p.y - this.start.y));
    } else if (this.mode === 'rotate' && this.f0) {
      const f = this.f0;
      let ang = Math.atan2(p.y - f.cy, p.x - f.cx) - Math.atan2(this.start.y - f.cy, this.start.x - f.cx);
      if (ev.shiftKey) ang = Math.round(ang / (Math.PI / 12)) * (Math.PI / 12);
      this.m = matRotateAbout(ang, f.cx, f.cy);
    } else if (this.mode === 'scale' && this.f0 && this.handle) {
      const f = this.f0, { hx, hy } = this.handle;
      const c = Math.cos(f.rot), s = Math.sin(f.rot);
      const dx = p.x - f.cx, dy = p.y - f.cy;
      const lx = dx * c + dy * s, ly = -dx * s + dy * c;
      const ax = (-hx * f.w) / 2, ay = (-hy * f.h) / 2;
      let sx = hx && f.w > 1e-6 ? (lx - ax) / (hx * f.w) : 1;
      let sy = hy && f.h > 1e-6 ? (ly - ay) / (hy * f.h) : 1;
      if (hx && hy && !ev.shiftKey) sx = sy = Math.max(sx, sy); // corners keep the aspect ratio
      sx = Math.max(0.03, sx); sy = Math.max(0.03, sy);
      const wax = f.cx + ax * c - ay * s, way = f.cy + ax * s + ay * c;
      const R: Mat = [c, s, -s, c, 0, 0], Ri: Mat = [c, -s, s, c, 0, 0];
      this.m = matMul(matTranslate(wax, way), matMul(R, matMul([sx, 0, 0, sy, 0, 0], matMul(Ri, matTranslate(-wax, -way)))));
    }
    if (this.mode === 'move' || this.mode === 'scale' || this.mode === 'rotate') {
      if (this.ids.size) {
        this.input.transformingSelection = true;
        e.ui.set({ selBusy: true }); // the floating action bar steps aside while the selection is dragged
        e.setPreview(this.ids, this.m);
      }
    }
    e.invalidateLive();
  }

  up(ev: PointerEvent) {
    const e = this.e, page = e.page;
    this.input.transformingSelection = false;
    e.ui.set({ selBusy: false });
    if (this.mode === 'move' || this.mode === 'scale' || this.mode === 'rotate') {
      if (this.moved && this.ids.size) {
        if (this.mode === 'move') {
          if (this.m[4] || this.m[5]) e.commit([{ kind: 'move', ids: [...this.ids], dx: this.m[4], dy: this.m[5] }]);
        } else {
          const ops: OpBody[] = [];
          for (const id of this.ids) {
            const o = page.objects[id];
            if (o) ops.push({ kind: 'update', id, patch: transformObject(o, this.m) });
          }
          e.commit(ops);
        }
      } else if (!this.moved && this.clicked) {
        if (this.clicked.type === 'graph' && this.onlySel === this.clicked.id) {
          e.setPreview(null);
          e.probeGraph(this.start.x, this.start.y, this.touch ? 18 : 10);
          return;
        }
        // double-click / double-tap a text box to edit it
        const now = performance.now();
        const last = this.input.lastTap;
        if (last && last.id === this.clicked.id && now - last.t < 450 && this.clicked.type === 'text') {
          this.input.lastTap = null;
          e.setPreview(null);
          e.beginText(this.clicked);
          return;
        }
        this.input.lastTap = { id: this.clicked.id, t: now };
      }
      e.setPreview(null);
      e.selChanged();
      return;
    }
    // lasso / rectangle
    const tiny = Math.hypot(this.cur.sx - this.start.sx, this.cur.sy - this.start.sy) < 4 && this.path.length < 12;
    if (!this.moved || tiny) {
      // plain click on empty space: a locked object under the cursor can still be selected (to unlock it)
      const locked = this.hitTop(true);
      e.setSelection(locked ? [locked.id] : []);
      return;
    }
    const found: string[] = ev.shiftKey ? [...e.sel] : [];
    let rect: BBox | null = null;
    if (this.mode === 'rect') {
      rect = [Math.min(this.start.x, this.cur.x), Math.min(this.start.y, this.cur.y), Math.max(this.start.x, this.cur.x), Math.max(this.start.y, this.cur.y)];
    }
    for (const id of page.order) {
      const o = page.objects[id];
      if (!o || o.locked) continue; // locked objects ignore sweeps, so ink over a locked image is easy to grab
      if (rect ? objInRect(o, rect) : objInLasso(o, this.path)) found.push(id);
    }
    e.setSelection(found);
  }

  cancel() {
    this.input.transformingSelection = false;
    this.e.ui.set({ selBusy: false });
    this.e.setPreview(null);
    this.e.invalidateLive();
  }

  draw(ctx: CanvasRenderingContext2D) {
    if (!this.moved || (this.mode !== 'lasso' && this.mode !== 'rect')) return;
    const z = this.e.cam.zoom;
    ctx.save();
    ctx.strokeStyle = '#1971c2';
    ctx.fillStyle = 'rgba(25,113,194,0.07)';
    ctx.lineWidth = 1.5 / z;
    ctx.setLineDash([6 / z, 5 / z]);
    ctx.beginPath();
    if (this.mode === 'rect') ctx.rect(this.start.x, this.start.y, this.cur.x - this.start.x, this.cur.y - this.start.y);
    else {
      ctx.moveTo(this.path[0], this.path[1]);
      for (let i = 2; i < this.path.length; i += 2) ctx.lineTo(this.path[i], this.path[i + 1]);
      ctx.closePath();
    }
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

// =====================================================================
// Shapes
// =====================================================================
function snapGrid(e: Engine, x: number, y: number): [number, number] {
  const u = gridUnit(e.page.background);
  if (!u || !e.settings.shape.snap) return [round1(x), round1(y)];
  return [Math.round(x / u) * u, Math.round(y / u) * u];
}

export function buildShape(kind: ShapeTool, ax: number, ay: number, bx: number, by: number, constrain: boolean): { kind: Shape['kind']; points: number[] } {
  if (kind === 'line' || kind === 'arrow' || kind === 'doubleArrow') {
    if (constrain) [bx, by] = snapTo15(ax, ay, bx, by);
    return { kind, points: [ax, ay, round1(bx), round1(by)] };
  }
  if (constrain) {
    const d = Math.max(Math.abs(bx - ax), Math.abs(by - ay));
    bx = ax + Math.sign(bx - ax || 1) * d;
    by = ay + Math.sign(by - ay || 1) * d;
  }
  if (kind === 'rect' || kind === 'ellipse') {
    return { kind, points: [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)] };
  }
  const top = Math.min(ay, by), bottom = Math.max(ay, by);
  if (kind === 'triangle') return { kind: 'triangle', points: [(ax + bx) / 2, top, bx, bottom, ax, bottom] };
  // right triangle: the right angle sits under the starting corner
  return { kind: 'triangle', points: [ax, top, ax, bottom, bx, bottom] };
}

class ShapeGesture implements Gesture {
  kind = 'shape';
  private a: [number, number];
  private shape: Shape | null = null;

  constructor(private e: Engine, p: Pt) {
    this.a = snapGrid(e, p.x, p.y);
  }

  move(pts: Pt[], ev: PointerEvent) {
    const e = this.e, p = pts[pts.length - 1];
    const [bx, by] = snapGrid(e, p.x, p.y);
    const s = e.settings.shape, pen = e.activeFav;
    const built = buildShape(s.kind, this.a[0], this.a[1], bx, by, ev.shiftKey);
    this.shape = {
      id: this.shape?.id ?? nanoid(12), type: 'shape', authorId: e.me.userId, createdAt: Date.now(), ...built,
      color: pen.color, size: pen.pen === 'highlighter' ? Math.min(pen.size, 6) : pen.size,
      ...(s.dashed ? { dashed: true } : {}),
      ...(s.fill && built.kind !== 'line' && built.kind !== 'arrow' && built.kind !== 'doubleArrow' ? { fill: s.fill, fillOpacity: s.fillOpacity } : {}),
    };
    e.invalidateLive();
  }

  up() {
    const sh = this.shape;
    if (!sh) return;
    const b = objBounds(sh);
    if ((b[2] - b[0] + (b[3] - b[1]) - sh.size * 2) * this.e.cam.zoom < 6) return; // just a click
    this.e.commit([{ kind: 'add', obj: sh }]);
  }
  cancel() { this.shape = null; }
  draw(ctx: CanvasRenderingContext2D) { if (this.shape) drawShape(ctx, this.shape, this.e.env); }
}

// =====================================================================
// Laser, insert space, box insertion, pan, ruler, graph interaction
// =====================================================================
class LaserGesture implements Gesture {
  kind = 'laser';
  private unsent: number[] = [];
  constructor(private e: Engine, p: Pt) { this.push(p); }
  private push(p: Pt) {
    this.e.laserPoint(p.x, p.y, this.e.settings.laser.mode);
    this.unsent.push(round1(p.x), round1(p.y));
  }
  move(pts: Pt[]) { for (const p of pts) this.push(p); }
  frame() {
    if (!this.unsent.length) return;
    const mode = this.e.settings.laser.mode;
    this.e.sync.emit('laser', { pageId: this.e.pageId, points: mode === 'dot' ? this.unsent.slice(-2) : this.unsent, mode });
    this.unsent = [];
  }
  up() { this.frame(); }
  cancel() { /* trail simply fades */ }
}

class SpaceGesture implements Gesture {
  kind = 'space';
  private axis: 'x' | 'y';
  private at: number;
  private delta = 0;
  private ids: Set<string>;
  constructor(private e: Engine, p: Pt) {
    this.axis = e.settings.space.axis;
    this.at = this.axis === 'y' ? p.y : p.x;
    this.ids = new Set(spaceTargets(e.page, this.axis, this.at));
  }
  move(pts: Pt[]) {
    const p = pts[pts.length - 1];
    this.delta = round1((this.axis === 'y' ? p.y : p.x) - this.at);
    if (this.ids.size) this.e.setPreview(this.ids, this.axis === 'y' ? matTranslate(0, this.delta) : matTranslate(this.delta, 0));
    this.e.invalidateLive();
  }
  up() {
    const e = this.e;
    if (this.ids.size && Math.abs(this.delta) * e.cam.zoom >= 2) {
      e.commit([{ kind: 'move', ids: [...this.ids], dx: this.axis === 'x' ? this.delta : 0, dy: this.axis === 'y' ? this.delta : 0 }]);
    }
    e.setPreview(null);
  }
  cancel() { this.e.setPreview(null); }
  draw(ctx: CanvasRenderingContext2D) {
    const v = this.e.viewBox(), z = this.e.cam.zoom;
    ctx.save();
    ctx.fillStyle = this.delta >= 0 ? 'rgba(25,113,194,0.10)' : 'rgba(224,49,49,0.10)';
    ctx.strokeStyle = '#1971c2';
    ctx.lineWidth = 1.5 / z;
    const a = this.at, b = this.at + this.delta;
    ctx.beginPath();
    if (this.axis === 'y') {
      ctx.fillRect(v[0], Math.min(a, b), v[2] - v[0], Math.abs(b - a));
      ctx.moveTo(v[0], a); ctx.lineTo(v[2], a);
      ctx.setLineDash([8 / z, 6 / z]);
      ctx.moveTo(v[0], b); ctx.lineTo(v[2], b);
    } else {
      ctx.fillRect(Math.min(a, b), v[1], Math.abs(b - a), v[3] - v[1]);
      ctx.moveTo(a, v[1]); ctx.lineTo(a, v[3]);
      ctx.setLineDash([8 / z, 6 / z]);
      ctx.moveTo(b, v[1]); ctx.lineTo(b, v[3]);
    }
    ctx.stroke();
    ctx.restore();
  }
}

class BoxGesture implements Gesture {
  kind = 'box';
  private a: [number, number];
  private b: [number, number];
  constructor(private e: Engine, private what: 'graph' | 'axes' | 'tape', p: Pt) {
    this.a = what === 'tape' ? [round1(p.x), round1(p.y)] : snapGrid(e, p.x, p.y);
    this.b = this.a;
  }
  move(pts: Pt[]) {
    const p = pts[pts.length - 1];
    this.b = this.what === 'tape' ? [round1(p.x), round1(p.y)] : snapGrid(this.e, p.x, p.y);
    this.e.invalidateLive();
  }
  private rect() {
    return { x: Math.min(this.a[0], this.b[0]), y: Math.min(this.a[1], this.b[1]), w: Math.abs(this.b[0] - this.a[0]), h: Math.abs(this.b[1] - this.a[1]) };
  }
  up() {
    const e = this.e;
    let { x, y, w, h } = this.rect();
    const base = { id: nanoid(12), authorId: e.me.userId, createdAt: Date.now() };
    const dragged = Math.max(w, h) * e.cam.zoom >= 12;
    if (this.what === 'tape') {
      if (!dragged || Math.min(w, h) * e.cam.zoom < 4) return;
      const tape: Tape = { ...base, type: 'tape', x, y, w, h, color: e.settings.tape.color, revealed: false };
      e.commit([{ kind: 'add', obj: tape }]);
      return;
    }
    const unit = e.settings.shape.snap ? gridUnit(e.page.background) : 0;
    let obj: Graph | Axes;
    if (this.what === 'graph') {
      if (!dragged) { w = 440; h = 340; x -= w / 2; y -= h / 2; }
      w = Math.max(160, w); h = Math.max(120, h);
      // square units by default
      const xr = 10, yr = Math.max(1, Math.round(((xr * h) / w) * 2) / 2);
      obj = {
        ...base, type: 'graph', x, y, w, h, xMin: -xr, xMax: xr, yMin: -yr, yMax: yr, showGrid: true, showNumbers: true,
        expressions: [{ id: nanoid(6), text: '', color: nextGraphColor(0), visible: true }], sliders: [],
      };
    } else if (e.settings.axes.mode === 'numberLine') {
      if (!dragged) { w = unit ? unit * 12 : 480; x -= w / 2; }
      w = Math.max(120, w); h = 70; y -= dragged ? 0 : h / 2;
      obj = { ...base, type: 'axes', x, y, w, h, xMin: -5, xMax: 5, yMin: 0, yMax: 1, xStep: 1, yStep: 1, showNumbers: true, mode: 'numberLine' };
    } else {
      if (!dragged) { w = h = unit ? unit * 10 : 400; x -= w / 2; y -= h / 2; if (unit) [x, y] = snapGrid(e, x, y); }
      w = Math.max(80, w); h = Math.max(80, h);
      // with a grid background, one tick per grid cell so hand-plotted points land on grid lines
      const cell = unit || Math.min(w, h) / 10;
      const cx = Math.max(2, Math.round(w / cell)), cy = Math.max(2, Math.round(h / cell));
      const xMin = -Math.floor(cx / 2), yMin = -Math.floor(cy / 2);
      const stepFor = (n: number) => (n > 24 ? 5 : n > 14 ? 2 : 1);
      obj = { ...base, type: 'axes', x, y, w, h, xMin, xMax: xMin + cx, yMin, yMax: yMin + cy, xStep: stepFor(cx), yStep: stepFor(cy), showNumbers: true, mode: 'xy' };
    }
    e.commit([{ kind: 'add', obj }]);
    e.setTool('select');
    e.setSelection([obj.id]);
  }
  cancel() { this.b = this.a; }
  draw(ctx: CanvasRenderingContext2D) {
    const { x, y, w, h } = this.rect();
    if (!w && !h) return;
    if (this.what === 'tape') { this.e.drawTapePreview(ctx, x, y, w, h, this.e.settings.tape.color); return; }
    const z = this.e.cam.zoom;
    ctx.save();
    ctx.strokeStyle = '#1971c2';
    ctx.fillStyle = 'rgba(25,113,194,0.06)';
    ctx.lineWidth = 1.5 / z;
    ctx.setLineDash([6 / z, 5 / z]);
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }
}

class PanGesture implements Gesture {
  kind = 'pan';
  private last: [number, number];
  constructor(private e: Engine, p: Pt) { this.last = [p.sx, p.sy]; }
  move(pts: Pt[]) {
    const p = pts[pts.length - 1];
    this.e.panBy(p.sx - this.last[0], p.sy - this.last[1]);
    this.last = [p.sx, p.sy];
  }
  up() { /* nothing */ }
  cancel() { /* nothing */ }
}

class RulerDragGesture implements Gesture {
  kind = 'ruler';
  private off: [number, number];
  constructor(private e: Engine, p: Pt) {
    const r = e.ui.get().ruler;
    this.off = [p.x - r.x, p.y - r.y];
  }
  move(pts: Pt[]) {
    const p = pts[pts.length - 1];
    this.e.setRuler({ x: round1(p.x - this.off[0]), y: round1(p.y - this.off[1]) });
  }
  up() { /* nothing */ }
  cancel() { /* nothing */ }
}

class RulerRotateGesture implements Gesture {
  kind = 'ruler';
  private a0: number;
  private angle0: number;
  constructor(private e: Engine, p: Pt) {
    const r = e.ui.get().ruler;
    this.a0 = Math.atan2(p.y - r.y, p.x - r.x);
    this.angle0 = r.angle;
  }
  move(pts: Pt[], ev: PointerEvent) {
    const p = pts[pts.length - 1], r = this.e.ui.get().ruler;
    const d = ((Math.atan2(p.y - r.y, p.x - r.x) - this.a0) * 180) / Math.PI;
    this.e.setRuler({ angle: snapAngle(this.angle0 + d, ev.shiftKey) });
  }
  up() { /* nothing */ }
  cancel() { /* nothing */ }
}

/** "Interact" mode: drag inside a selected graph to pan its window. */
class GraphPanGesture implements Gesture {
  kind = 'graph';
  private win: [number, number, number, number];
  private pending: Record<string, number> | null = null;
  private moved = false;
  constructor(private e: Engine, private g: Graph, private start: Pt) {
    this.win = [g.xMin, g.xMax, g.yMin, g.yMax];
  }
  move(pts: Pt[]) {
    const p = pts[pts.length - 1], g = this.g;
    if (!this.moved && Math.hypot(p.sx - this.start.sx, p.sy - this.start.sy) < 4) return;
    this.moved = true;
    const rot = -(g.rotation || 0);
    const wx = p.x - this.start.x, wy = p.y - this.start.y;
    const lx = wx * Math.cos(rot) - wy * Math.sin(rot), ly = wx * Math.sin(rot) + wy * Math.cos(rot);
    const dx = (-lx / g.w) * (this.win[1] - this.win[0]), dy = (ly / g.h) * (this.win[3] - this.win[2]);
    this.pending = { xMin: this.win[0] + dx, xMax: this.win[1] + dx, yMin: this.win[2] + dy, yMax: this.win[3] + dy };
  }
  frame() {
    if (!this.pending) return;
    this.e.commit([{ kind: 'update', id: this.g.id, patch: this.pending }], { undo: false });
    this.pending = null;
  }
  up() {
    this.frame();
    if (!this.moved) this.e.probeGraph(this.start.x, this.start.y, 12);
  }
  cancel() { /* keep what was applied */ }
}

// =====================================================================
// Crop: drag the handles (or the inside) of the crop rectangle
// =====================================================================
class CropGesture implements Gesture {
  kind = 'crop';
  private start: Crop;
  private h: { hx: number; hy: number } | null = null;
  private u0: number; private v0: number;
  constructor(private e: Engine, private o: ImageObj, p: Pt, ev: PointerEvent) {
    this.start = { ...e.ui.get().crop!.rect };
    const grab = ev.pointerType === 'touch' ? 22 : 12;
    for (const h of e.cropHandles(o, this.start)) {
      const [x, y] = e.toScreen(h.x, h.y);
      if (Math.hypot(x - p.sx, y - p.sy) <= grab) { this.h = h; break; }
    }
    [this.u0, this.v0] = toImageFraction(o, p.x, p.y);
    const r = this.start;
    const inside = this.u0 >= r.x && this.u0 <= r.x + r.w && this.v0 >= r.y && this.v0 <= r.y + r.h;
    if (!this.h && inside) this.h = { hx: 0, hy: 0 }; // drag the whole crop
  }
  move(pts: Pt[]) {
    if (!this.h) return;
    const p = pts[pts.length - 1];
    const [u, v] = toImageFraction(this.o, p.x, p.y);
    const rect = dragCrop(this.start, this.h.hx, this.h.hy, u, v, u - this.u0, v - this.v0);
    this.e.ui.set((st) => (st.crop ? { crop: { ...st.crop, rect } } : {}));
    this.e.invalidateLive();
  }
  up() { /* the crop is applied with Done */ }
  cancel() { this.e.ui.set((st) => (st.crop ? { crop: { ...st.crop, rect: this.start } } : {})); this.e.invalidateLive(); }
}

// =====================================================================
// Input: pointer, wheel, keyboard, clipboard, drag & drop
// =====================================================================
export class Input {
  gesture: Gesture | null = null;
  transformingSelection = false;
  hoverWorld: [number, number] | null = null;
  lastTap: { id: string; t: number } | null = null;
  sawPen = false;

  private pointerId = -1;
  private gestureTouch = false;
  private down: { sx: number; sy: number; t: number; tape: Tape | null } | null = null;
  private hover: { sx: number; sy: number } | null = null;
  private touches = new Map<number, { sx: number; sy: number }>();
  private pinch: { d0: number; a0: number; zoom0: number; wx: number; wy: number; ruler: { angle: number; x: number; y: number; cx: number; cy: number } | null } | null = null;
  private spaceDown = false;
  private laserKey: { t: number } | null = null;
  /** two-finger tap = undo, three-finger tap = redo */
  private taps = new MultiTapDetector();
  private lastPenAt = -1e9;
  private cursorDirty = false;
  private cursorSentAt = 0;
  private rulerRaw: number | null = null;
  private safariZoom0 = 1;
  private el: HTMLCanvasElement;

  constructor(private e: Engine) {
    this.el = e.live;
    const el = this.el;
    el.addEventListener('pointerdown', this.onDown);
    el.addEventListener('pointermove', this.onMove);
    el.addEventListener('pointerup', this.onUp);
    el.addEventListener('pointercancel', this.onCancel);
    el.addEventListener('pointerleave', this.onLeave);
    el.addEventListener('wheel', this.onWheel, { passive: false });
    el.addEventListener('contextmenu', this.prevent);
    el.addEventListener('dragstart', this.prevent);
    el.addEventListener('selectstart', this.prevent);
    el.addEventListener('gesturestart', this.onGestureStart as EventListener);
    el.addEventListener('gesturechange', this.onGestureChange as EventListener);
    el.addEventListener('gestureend', this.prevent);
    el.addEventListener('dragover', this.onDragOver);
    el.addEventListener('drop', this.onDrop);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    document.addEventListener('copy', this.onCopy);
    document.addEventListener('cut', this.onCut);
    document.addEventListener('paste', this.onPaste);
  }

  destroy() {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    document.removeEventListener('copy', this.onCopy);
    document.removeEventListener('cut', this.onCut);
    document.removeEventListener('paste', this.onPaste);
  }

  private prevent = (ev: Event) => ev.preventDefault();

  private pt(ev: PointerEvent | MouseEvent, rect = this.el.getBoundingClientRect()): Pt {
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [x, y] = this.e.toWorld(sx, sy);
    return { x, y, sx, sy, p: (ev as PointerEvent).pressure ?? 0.5 };
  }

  cancelGesture() {
    if (!this.gesture) return;
    const g = this.gesture;
    this.gesture = null;
    try { g.cancel(); } catch (err) { console.error(err); }
    try { this.el.releasePointerCapture(this.pointerId); } catch { /* not captured */ }
    this.pointerId = -1;
    this.down = null;
    this.e.invalidateLive();
    this.e.gestureEnded();
  }

  // ---------- pointer ----------
  private onDown = (ev: PointerEvent) => {
    ev.preventDefault();
    const e = this.e;
    const ui = e.ui.get();
    // clicking the board closes popovers and commits any text being edited
    const active = document.activeElement as HTMLElement | null;
    if (active && active !== document.body && isEditable(active)) active.blur();
    if (ui.popover) e.ui.set({ popover: null });
    const te = e.ui.get().textEdit;
    if (te) e.commitText(te.text);
    e.stopFollow();
    if (ev.pointerType === 'pen') { this.sawPen = true; this.lastPenAt = performance.now(); this.taps.spoil(); }
    const p = this.pt(ev);

    if (ev.pointerType === 'touch') {
      this.hover = null;
      this.touches.set(ev.pointerId, { sx: p.sx, sy: p.sy });
      this.taps.start(ev.pointerId, p.sx, p.sy, performance.now());
      // a palm resting while the pen writes is not a gesture
      if (this.gesture && !this.gestureTouch) this.taps.spoil();
      if (this.touches.size === 2) {
        if (this.gesture && this.gestureTouch) this.cancelGesture();
        this.startPinch();
        return;
      }
      if (this.touches.size > 2) return;
      // palm rejection: once a pen has been seen (or unless "draw with finger" is on), touch only navigates
      if (this.sawPen || !e.settings.drawWithFinger) {
        if (this.gesture) return;
        const r = ui.ruler;
        const g = r.visible && Math.hypot(p.x - knobPos(r)[0], p.y - knobPos(r)[1]) <= 22 / e.cam.zoom ? new RulerRotateGesture(e, p)
          : r.visible && rulerContains(r, p.x, p.y) ? new RulerDragGesture(e, p) : new PanGesture(e, p);
        this.begin(g, ev, p, true);
        return;
      }
    }
    if (this.gesture) return;
    const g = this.pick(ev, p);
    if (g) this.begin(g, ev, p, ev.pointerType === 'touch');
  };

  private begin(g: Gesture, ev: PointerEvent, p: Pt, touch: boolean) {
    this.gesture = g;
    this.gestureTouch = touch;
    this.pointerId = ev.pointerId;
    const tool = this.activeTool();
    // a tap on a tape reveals / hides it instead of doing the tool's normal thing
    let tape: Tape | null = null;
    if (g.kind !== 'ruler' && g.kind !== 'eraser' && g.kind !== 'crop') {
      const t = this.tapeAt(p);
      if (t && (tool === 'select' || tool === 'tape' || tool === 'laser' || g.kind === 'pan' || ((tool === 'pen' || tool === 'highlighter') && !t.revealed))) tape = t;
    }
    this.down = { sx: p.sx, sy: p.sy, t: performance.now(), tape };
    try { this.el.setPointerCapture(ev.pointerId); } catch { /* ignore */ }
    this.cursorDirty = true;
    this.e.invalidateLive();
  }

  private tapeAt(p: Pt): Tape | null {
    const page = this.e.page;
    for (let i = page.order.length - 1; i >= 0; i--) {
      const o = page.objects[page.order[i]];
      if (o?.type === 'tape' && hitObject(o, p.x, p.y, 0)) return o;
    }
    return null;
  }

  /** Decides what a pointer-down does. */
  private pick(ev: PointerEvent, p: Pt): Gesture | null {
    const e = this.e, ui = e.ui.get(), z = e.cam.zoom;
    const mouse = ev.pointerType === 'mouse', pen = ev.pointerType === 'pen';
    if (mouse && (ev.button === 1 || (ev.button === 0 && this.spaceDown))) return new PanGesture(e, p);
    if (mouse && ev.button !== 0) return null;
    if (e.replay) return new PanGesture(e, p);
    if (!pen && this.spaceDown) return new PanGesture(e, p);
    // pen hardware: eraser end erases, barrel button lassos
    if (pen && ((ev.buttons & 32) || ev.button === 5)) return new EraserGesture(e, p);
    if (pen && ((ev.buttons & 2) || ev.button === 2)) return new SelectGesture(e, this, p, ev);

    const crop = e.cropTarget();
    if (crop) return new CropGesture(e, crop, p, ev);
    const tool = this.activeTool();
    const penLike = tool === 'pen' || tool === 'highlighter';
    const r = ui.ruler;
    if (r.visible) {
      const k = knobPos(r);
      if (Math.hypot(p.x - k[0], p.y - k[1]) <= 17 / z) { this.rulerRaw = null; return new RulerRotateGesture(e, p); }
      if (penLike) {
        const snap = findSnap(r, p.x, p.y, 14 / z);
        if (snap) { if (e.sel.size) e.setSelection([]); return new PenGesture(e, tool, p, ev, snap); }
      }
      if (rulerContains(r, p.x, p.y)) return new RulerDragGesture(e, p);
    }
    if (ui.graphInteract && ui.sel?.single?.type === 'graph') {
      const g = e.page.objects[ui.sel.single.id];
      if (g?.type === 'graph' && pointInFrame(frameOf(g)!, p.x, p.y)) return new GraphPanGesture(e, g, p);
    }
    // a selection made with the barrel button can be dragged with any tool
    if (tool !== 'select' && !ui.heldTool && e.sel.size) {
      const f = e.selFrame();
      const onHandle = f && !e.selLocked() && e.handles(f).some((h) => Math.hypot(p.x - h.x, p.y - h.y) <= 11 / z);
      if (f && !e.selLocked() && (onHandle || pointInFrame(f, p.x, p.y, 2 / z))) return new SelectGesture(e, this, p, ev);
      e.setSelection([]);
    }
    switch (tool) {
      case 'pen': case 'highlighter': return new PenGesture(e, tool, p, ev, null);
      case 'eraser': return new EraserGesture(e, p);
      case 'select': return new SelectGesture(e, this, p, ev);
      case 'shape': return new ShapeGesture(e, p);
      case 'laser': return new LaserGesture(e, p);
      case 'space': return new SpaceGesture(e, p);
      case 'graph': case 'axes': case 'tape': return new BoxGesture(e, tool, p);
      case 'text': {
        const page = e.page;
        for (let i = page.order.length - 1; i >= 0; i--) {
          const o = page.objects[page.order[i]];
          if (o?.type === 'text' && !o.locked && hitObject(o, p.x, p.y, 4 / z)) { e.beginText(o); return null; }
        }
        e.beginText({ x: round1(p.x), y: round1(p.y - e.settings.text.fontSize * 0.65) });
        return null;
      }
    }
    return null;
  }

  private onMove = (ev: PointerEvent) => {
    const rect = this.el.getBoundingClientRect();
    if (ev.pointerType === 'pen') this.lastPenAt = performance.now();
    if (ev.pointerType === 'touch' && this.touches.has(ev.pointerId)) {
      const p = this.pt(ev, rect);
      this.touches.set(ev.pointerId, { sx: p.sx, sy: p.sy });
      this.taps.move(ev.pointerId, p.sx, p.sy);
      if (this.pinch && this.touches.size >= 2) { this.movePinch(); return; }
    }
    if (this.gesture && ev.pointerId === this.pointerId) {
      // coalesced events keep fast strokes and tiny handwriting accurate
      const raw = typeof ev.getCoalescedEvents === 'function' ? ev.getCoalescedEvents() : [];
      const list = raw.length ? raw : [ev];
      const pts = list.map((c) => this.pt(c, rect));
      try { this.gesture.move(pts, ev); } catch (err) { console.error('[input] move', err); }
      const last = pts[pts.length - 1];
      this.track(ev, last);
      return;
    }
    if (ev.pointerType !== 'touch') this.track(ev, this.pt(ev, rect));
  };

  private track(ev: PointerEvent, p: Pt) {
    if (ev.pointerType !== 'touch') this.hover = { sx: p.sx, sy: p.sy };
    this.hoverWorld = [p.x, p.y];
    this.cursorDirty = true;
    this.e.invalidateLive();
  }

  private onUp = (ev: PointerEvent) => {
    if (ev.pointerType === 'pen') this.lastPenAt = performance.now();
    if (ev.pointerType === 'touch') {
      this.touches.delete(ev.pointerId);
      if (this.touches.size < 2) this.pinch = null;
      const tap = this.taps.end(ev.pointerId, performance.now());
      if (tap && performance.now() - this.lastPenAt > 500) this.multiTap(tap);
    }
    if (!this.gesture || ev.pointerId !== this.pointerId) return;
    const g = this.gesture, d = this.down;
    this.gesture = null;
    this.pointerId = -1;
    this.down = null;
    try { this.el.releasePointerCapture(ev.pointerId); } catch { /* ignore */ }
    const p = this.pt(ev);
    try {
      const tap = d && d.tape && Math.hypot(p.sx - d.sx, p.sy - d.sy) < 6 && performance.now() - d.t < 600;
      if (tap) {
        g.cancel();
        const cur = this.e.page.objects[d!.tape!.id];
        if (cur?.type === 'tape') this.e.commit([{ kind: 'update', id: cur.id, patch: { revealed: !cur.revealed } }], { undo: false });
      } else g.up(ev);
    } catch (err) {
      console.error('[input] up', err);
    }
    this.e.invalidateLive();
    this.e.gestureEnded();
  };

  /** Two-finger tap: undo. Three-finger tap: redo. Nothing is drawn and the view does not move. */
  private multiTap(kind: 'undo' | 'redo') {
    const e = this.e;
    if (this.gesture) this.cancelGesture();
    if (e.replay || e.ui.get().crop) return;
    const s = e.ui.get();
    if (kind === 'undo' ? !s.canUndo : !s.canRedo) { e.toast(kind === 'undo' ? 'Nothing to undo' : 'Nothing to redo', 'info', { ms: 1200 }); return; }
    if (kind === 'undo') e.undo(); else e.redo();
    e.toast(kind === 'undo' ? 'Undo' : 'Redo', 'info', { ms: 900 });
  }

  private onCancel = (ev: PointerEvent) => {
    if (ev.pointerType === 'touch') {
      this.touches.delete(ev.pointerId);
      if (this.touches.size < 2) this.pinch = null;
      this.taps.cancel(ev.pointerId);
    }
    if (this.gesture && ev.pointerId === this.pointerId) {
      // a stroke interrupted by the system is kept rather than lost
      const g = this.gesture;
      this.gesture = null;
      this.pointerId = -1;
      this.down = null;
      try { if (g.kind === 'pen') g.up(ev); else g.cancel(); } catch (err) { console.error(err); }
      this.e.invalidateLive();
      this.e.gestureEnded();
    }
  };

  private onLeave = (ev: PointerEvent) => {
    if (this.gesture && ev.pointerId === this.pointerId) return;
    this.hover = null;
    this.hoverWorld = null;
    this.e.invalidateLive();
  };

  // ---------- two-finger pinch / pan (and two-finger ruler rotation) ----------
  private twoTouches() {
    const [a, b] = [...this.touches.values()];
    return { a, b, cx: (a.sx + b.sx) / 2, cy: (a.sy + b.sy) / 2, d: Math.hypot(b.sx - a.sx, b.sy - a.sy) || 1, ang: Math.atan2(b.sy - a.sy, b.sx - a.sx) };
  }

  private startPinch() {
    const e = this.e, t = this.twoTouches();
    const [wx, wy] = e.toWorld(t.cx, t.cy);
    const r = e.ui.get().ruler;
    const [ax, ay] = e.toWorld(t.a.sx, t.a.sy), [bx, by] = e.toWorld(t.b.sx, t.b.sy);
    const onRuler = r.visible && rulerContains(r, ax, ay) && rulerContains(r, bx, by);
    this.pinch = { d0: t.d, a0: t.ang, zoom0: e.cam.zoom, wx, wy, ruler: onRuler ? { angle: r.angle, x: r.x, y: r.y, cx: wx, cy: wy } : null };
  }

  private movePinch() {
    const e = this.e, pn = this.pinch!, t = this.twoTouches();
    if (pn.ruler) {
      const [wx, wy] = e.toWorld(t.cx, t.cy);
      const deg = ((t.ang - pn.a0) * 180) / Math.PI;
      e.setRuler({ angle: snapAngle(pn.ruler.angle + deg, false), x: round1(pn.ruler.x + wx - pn.ruler.cx), y: round1(pn.ruler.y + wy - pn.ruler.cy) });
      return;
    }
    const zoom = Math.max(0.1, Math.min(8, pn.zoom0 * (t.d / pn.d0)));
    e.setCamera({ x: pn.wx - t.cx / zoom, y: pn.wy - t.cy / zoom, zoom });
  }

  // ---------- wheel ----------
  private onWheel = (ev: WheelEvent) => {
    ev.preventDefault();
    const e = this.e, ui = e.ui.get();
    const unit = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? e.vh : 1;
    let dx = ev.deltaX * unit, dy = ev.deltaY * unit;
    const p = this.pt(ev);
    if (ev.altKey && ui.ruler.visible) {
      const d = dy || dx;
      if (!d) return;
      if (this.rulerRaw === null) this.rulerRaw = ui.ruler.angle;
      this.rulerRaw += ev.shiftKey ? Math.sign(d) * 5 : Math.max(-6, Math.min(6, d * 0.03));
      e.setRuler({ angle: snapAngle(this.rulerRaw, ev.shiftKey, 1.6) });
      return;
    }
    const g = ui.graphInteract && ui.sel?.single?.type === 'graph' ? e.page.objects[ui.sel.single.id] : null;
    if (g?.type === 'graph' && pointInFrame(frameOf(g)!, p.x, p.y)) {
      // zoom the graph window about the cursor
      const k = Math.exp(Math.max(-1, Math.min(1, dy * (Math.abs(dy) >= 50 ? 0.002 : 0.01))));
      const rot = -(g.rotation || 0), f = frameOf(g)!;
      const lx = (p.x - f.cx) * Math.cos(rot) - (p.y - f.cy) * Math.sin(rot), ly = (p.x - f.cx) * Math.sin(rot) + (p.y - f.cy) * Math.cos(rot);
      const gx = g.xMin + (lx / g.w + 0.5) * (g.xMax - g.xMin), gy = g.yMax - (ly / g.h + 0.5) * (g.yMax - g.yMin);
      const span = (g.xMax - g.xMin) * k;
      if (span > 1e-6 && span < 1e7) {
        e.commit([{ kind: 'update', id: g.id, patch: { xMin: gx + (g.xMin - gx) * k, xMax: gx + (g.xMax - gx) * k, yMin: gy + (g.yMin - gy) * k, yMax: gy + (g.yMax - gy) * k } }], { undo: false });
      }
      return;
    }
    e.stopFollow();
    if (ev.ctrlKey || ev.metaKey) {
      // ctrl+wheel and trackpad pinch (which arrives as ctrl+wheel with small deltas)
      e.zoomAt(p.sx, p.sy, Math.exp(-dy * (Math.abs(dy) >= 50 ? 0.0022 : 0.011)));
      return;
    }
    if (ev.shiftKey && !dx) { dx = dy; dy = 0; }
    e.panBy(-dx, -dy);
  };

  private onGestureStart = (ev: Event) => { ev.preventDefault(); this.safariZoom0 = this.e.cam.zoom; };
  private onGestureChange = (ev: Event & { scale?: number; clientX?: number; clientY?: number }) => {
    ev.preventDefault();
    if (this.touches.size) return; // touch pinch is handled by pointer events
    const rect = this.el.getBoundingClientRect();
    const target = this.safariZoom0 * (ev.scale ?? 1);
    this.e.zoomAt((ev.clientX ?? rect.width / 2) - rect.left, (ev.clientY ?? rect.height / 2) - rect.top, target / this.e.cam.zoom);
  };

  // ---------- per-frame work ----------
  /** Called once per animation frame; returns true while a gesture needs continuous frames. */
  frame(now: number): boolean {
    const g = this.gesture;
    if (g?.frame) {
      try { g.frame(now); } catch (err) { console.error('[input] frame', err); }
    }
    if (this.cursorDirty && now - this.cursorSentAt >= 34 && this.hoverWorld) {
      this.cursorDirty = false;
      this.cursorSentAt = now;
      this.e.sync.emit('cursor', { pageId: this.e.pageId, x: round1(this.hoverWorld[0]), y: round1(this.hoverWorld[1]), tool: this.activeTool() }, true);
    }
    return !!g || this.cursorDirty;
  }

  draw(ctx: CanvasRenderingContext2D) {
    try { this.gesture?.draw?.(ctx); } catch (err) { console.error('[input] draw', err); }
  }

  /** Local cursor: a dot in the pen's colour and size, or a circle showing the eraser area. */
  drawCursor(ctx: CanvasRenderingContext2D) {
    const e = this.e, ui = e.ui.get(), tool = this.activeTool();
    const kind = this.gesture?.kind;
    let css = 'default';
    const erasing = kind === 'eraser' || (!kind && tool === 'eraser');
    const penning = kind === 'pen' || (!kind && (tool === 'pen' || tool === 'highlighter'));
    if (kind === 'pan' || this.spaceDown) css = kind === 'pan' ? 'grabbing' : 'grab';
    else if (e.replay) css = 'grab';
    else if (kind === 'ruler') css = 'move';
    else if (erasing || penning || tool === 'laser') css = 'none';
    else if (tool === 'text') css = 'text';
    else if (tool === 'shape' || tool === 'graph' || tool === 'axes' || tool === 'tape') css = 'crosshair';
    else if (tool === 'space') css = e.settings.space.axis === 'y' ? 'row-resize' : 'col-resize';
    if (this.el.style.cursor !== css) this.el.style.cursor = css;
    const h = this.hover;
    if (!h || css !== 'none') return;
    ctx.save();
    if (erasing) {
      const r = ERASER_SIZES[e.settings.eraser.size];
      ctx.beginPath();
      ctx.arc(h.sx, h.sy, r, 0, Math.PI * 2);
      ctx.fillStyle = e.env.dark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.05)';
      ctx.fill();
      haloRing(ctx, h.sx, h.sy, r, 0.8);
    } else if (tool === 'laser' && !penning) {
      ctx.beginPath();
      ctx.arc(h.sx, h.sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = ui.me.color;
      ctx.fill();
      haloRing(ctx, h.sx, h.sy, 4.5, 0.6);
    } else {
      const st = e.currentStyle(tool === 'highlighter' ? 'highlighter' : 'pen', false);
      const r = Math.max(1.5, (st.size * e.cam.zoom) / 2);
      ctx.beginPath();
      ctx.arc(h.sx, h.sy, r, 0, Math.PI * 2);
      ctx.globalAlpha = st.pen === 'highlighter' ? 0.45 : this.gesture ? 0 : 0.9;
      ctx.fillStyle = resolveColor(st.color, e.env.dark);
      ctx.fill();
      ctx.globalAlpha = 1;
      // the ring keeps the dot visible over any background (white ink on a white PDF page, dark ink on a photo…)
      haloRing(ctx, h.sx, h.sy, Math.max(5, r + 1), this.gesture ? 0.35 : 0.6);
    }
    ctx.restore();
  }

  // ---------- keyboard ----------
  private onBlur = () => { this.spaceDown = false; this.releaseLaserKey(false); };
  private onKeyUp = (ev: KeyboardEvent) => {
    if (ev.code === 'Space') { this.spaceDown = false; this.e.invalidateLive(); }
    if (ev.key.toLowerCase() === 'l') this.releaseLaserKey(true);
  };

  /** How long L must be held for the laser to be temporary. */
  static readonly HOLD_MS = 250;

  /** L was released: a short press keeps the laser selected, a long one returns to the previous tool. */
  private releaseLaserKey(mayStay: boolean) {
    const k = this.laserKey;
    if (!k) return;
    this.laserKey = null;
    if (this.gesture?.kind === 'laser') this.cancelGesture();
    this.e.ui.set({ heldTool: null });
    if (mayStay && performance.now() - k.t <= Input.HOLD_MS) this.e.setTool('laser');
    this.e.invalidateLive();
  }

  /** The tool a pointer-down uses right now (a held key overrides the selected tool). */
  private activeTool() {
    const ui = this.e.ui.get();
    return ui.heldTool ?? ui.tool;
  }

  private onKeyDown = (ev: KeyboardEvent) => {
    if (isEditable(ev.target)) {
      // after dragging a slider, Ctrl+Z must still undo
      const k = ev.key.toLowerCase();
      if (!(isControl(ev.target) && (((ev.ctrlKey || ev.metaKey) && (k === 'z' || k === 'y')) || ev.key === 'Escape'))) return;
      (ev.target as HTMLElement).blur();
    }
    const e = this.e, ui = e.ui.get();
    const mod = ev.ctrlKey || ev.metaKey;
    const key = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
    const done = () => ev.preventDefault();

    if (ui.dialog || ui.restore) {
      if (key === 'Escape' && ui.dialog) { e.ui.set({ dialog: null }); done(); }
      return;
    }
    if (ui.crop && !mod) {
      if (key === 'Enter') { e.commitCrop(); done(); }
      else if (key === 'Escape') { if (this.gesture) this.cancelGesture(); else e.cancelCrop(); done(); }
      return;
    }
    if (mod) {
      switch (key) {
        case 'z': if (ev.shiftKey) e.redo(); else e.undo(); return done();
        case 'y': e.redo(); return done();
        case 'd': e.duplicateSelection(); return done();
        case 'a': e.selectAll(); return done();
        case '0': e.resetZoom(); return done();
        case '=': case '+': e.zoomStep(1); return done();
        case '-': e.zoomStep(-1); return done();
      }
      return; // copy / cut / paste arrive as clipboard events
    }
    if (ev.altKey) return;
    if (ev.code === 'Space') { this.spaceDown = true; e.invalidateLive(); return done(); }
    if (e.replay) {
      if (key === 'Escape') e.stopReplay();
      return;
    }
    if (key >= '1' && key <= '9' && !ev.shiftKey) { e.selectFav(Number(key) - 1); return done(); }
    switch (key) {
      case 'p':
        if (ev.shiftKey) { this.rulerRaw = null; e.toggleRuler('protractor'); } else e.setTool('pen');
        break;
      case 'h': e.setTool('highlighter'); break;
      case 'e':
        if (ui.tool === 'eraser') {
          e.setSettings((s) => ({ eraser: { ...s.eraser, mode: s.eraser.mode === 'stroke' ? 'partial' : 'stroke' } }));
          e.toast(e.settings.eraser.mode === 'stroke' ? 'Stroke eraser' : 'Partial eraser');
        } else e.setTool('eraser');
        break;
      case 'v': case 's': e.setTool('select'); break;
      case 'r': if (ev.shiftKey) e.setTool('shape'); else { this.rulerRaw = null; e.toggleRuler(); } break;
      case 'l':
        // a quick press switches to the laser as before; holding the key gives the laser only while it is held
        if (ev.repeat || this.laserKey) break;
        if (ui.tool === 'laser') break;
        this.cancelGesture();
        this.laserKey = { t: performance.now() };
        e.ui.set({ heldTool: 'laser', popover: null });
        e.invalidateLive();
        break;
      case 't': e.setTool('text'); break;
      case 'g': e.setTool('graph'); break;
      case 'k': e.setTool('tape'); break;
      case 'i': e.setTool('space'); break;
      case 'a':
        if (ui.tool === 'axes') {
          e.setSettings((s) => ({ axes: { mode: s.axes.mode === 'xy' ? 'numberLine' : 'xy' } }));
          e.toast(e.settings.axes.mode === 'xy' ? 'x–y axes' : 'Number line');
        } else e.setTool('axes');
        break;
      case 'f': e.ui.set({ focus: !ui.focus, popover: null }); break;
      case '[': e.nudgeThickness(-1); break;
      case ']': e.nudgeThickness(1); break;
      case '?': e.ui.set({ dialog: 'shortcuts' }); break;
      case 'Delete': case 'Backspace': e.deleteSelection(); break;
      case 'PageUp': e.stepPage(-1); break;
      case 'PageDown': e.stepPage(1); break;
      case 'Escape':
        if (this.gesture) this.cancelGesture();
        else if (ui.popover) e.ui.set({ popover: null });
        else if (e.sel.size) e.setSelection([]);
        else if (ui.following) e.stopFollow();
        else if (ui.focus) e.ui.set({ focus: false });
        break;
      case 'ArrowLeft': case 'ArrowRight': case 'ArrowUp': case 'ArrowDown': {
        const ids = e.selObjects().filter((o) => !o.locked).map((o) => o.id);
        if (!ids.length) return;
        const d = (ev.shiftKey ? 10 : 1) / e.cam.zoom;
        e.commit([{ kind: 'move', ids, dx: key === 'ArrowLeft' ? -d : key === 'ArrowRight' ? d : 0, dy: key === 'ArrowUp' ? -d : key === 'ArrowDown' ? d : 0 }]);
        break;
      }
      default: return;
    }
    done();
  };

  // ---------- clipboard, drag & drop ----------
  private onCopy = (ev: ClipboardEvent) => {
    if (isEditable(ev.target) || isEditable(document.activeElement)) return;
    const marker = this.e.copySelection();
    if (!marker) return;
    ev.clipboardData?.setData('text/plain', marker);
    ev.preventDefault();
  };

  private onCut = (ev: ClipboardEvent) => {
    if (isEditable(ev.target) || isEditable(document.activeElement)) return;
    const marker = this.e.copySelection();
    if (!marker) return;
    ev.clipboardData?.setData('text/plain', marker);
    ev.preventDefault();
    this.e.deleteSelection();
  };

  private onPaste = (ev: ClipboardEvent) => {
    if (isEditable(ev.target) || isEditable(document.activeElement)) return;
    const e = this.e, ui = e.ui.get();
    if (!ui.ready || ui.dialog || ui.restore || e.replay) return;
    const dt = ev.clipboardData;
    if (!dt) return;
    ev.preventDefault();
    const files: File[] = [];
    for (const item of Array.from(dt.items)) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const f = item.getAsFile();
        if (f) files.push(f);
      }
    }
    const text = dt.getData('text/plain');
    // board objects copied here, in another tab or on another board
    if (text.startsWith(CLIP_PREFIX)) { void e.pasteBoardText(text).catch((err) => e.toast(`Paste failed: ${(err as Error).message}`, 'error')); return; }
    // every image on the clipboard, not only the first
    if (files.length) { void e.insertImages(files); return; }
    if (text.trim()) e.insertTextAtCenter(text);
  };

  private onDragOver = (ev: DragEvent) => {
    ev.preventDefault();
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'copy';
  };

  private onDrop = (ev: DragEvent) => {
    ev.preventDefault();
    const files = Array.from(ev.dataTransfer?.files ?? []).filter((f) => f.type.startsWith('image/'));
    if (!files.length) { this.e.toast('Drop an image file (PNG, JPEG, GIF or WebP)'); return; }
    // the images land where they were dropped
    const p = this.pt(ev);
    void this.e.insertImages(files, { at: { x: p.x, y: p.y } });
  };
}
