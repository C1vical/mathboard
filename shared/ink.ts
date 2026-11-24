import { getStroke } from 'perfect-freehand';
import type { PenType } from './types';

export interface InkStyle {
  pen: PenType;
  size: number;
  stab?: number | null;
  sim?: boolean | null;
}

export const DEFAULT_STAB = 30;

function pathLength(points: number[], cap: number): number {
  let len = 0;
  for (let i = 3; i + 1 < points.length && len < cap; i += 3) {
    len += Math.hypot(points[i] - points[i - 3], points[i + 1] - points[i - 2]);
  }
  return len;
}

/**
 * perfect-freehand options for a pen type. The single source of truth: local live
 * ink, remote previews and committed strokes all go through this with `last: true`,
 * so a stroke never changes shape when it is committed.
 */
export function inkOptions(style: InkStyle, points: number[]) {
  const stab = Math.max(0, Math.min(100, style.stab ?? DEFAULT_STAB)) / 100;
  const base = {
    size: style.size,
    streamline: 0.08 + 0.72 * stab,
    smoothing: 0.35 + 0.5 * stab,
    simulatePressure: !!style.sim,
    last: true,
    easing: (t: number) => t,
  };
  switch (style.pen) {
    case 'fountain': {
      // taper scales with stroke length so dots and short marks stay visible
      const len = pathLength(points, style.size * 12);
      return {
        ...base, thinning: 0.7,
        start: { taper: Math.min(style.size * 1.5, len * 0.2), cap: true },
        end: { taper: Math.min(style.size * 3.5, len * 0.3), cap: true },
      };
    }
    case 'pencil': return { ...base, thinning: 0.35 };
    case 'marker': return { ...base, thinning: 0, simulatePressure: false };
    case 'highlighter': return { ...base, thinning: 0, simulatePressure: false };
    default: return { ...base, thinning: 0.18 };
  }
}

/** Outline polygon ([x,y] pairs) of a stroke. */
export function strokeOutline(style: InkStyle, points: number[]): number[][] {
  const n = Math.floor(points.length / 3);
  const input: number[][] = new Array(n);
  const flat = style.pen === 'marker' || style.pen === 'highlighter';
  for (let i = 0; i < n; i++) {
    input[i] = [points[i * 3], points[i * 3 + 1], flat || style.sim ? 0.5 : points[i * 3 + 2]];
  }
  return getStroke(input, inkOptions(style, points));
}

/** Pencil darkness from average pressure — deterministic so every client matches. */
export function pencilAlpha(style: InkStyle, points: number[]): number {
  if (style.sim) return 0.8;
  let sum = 0, n = 0;
  for (let i = 2; i < points.length; i += 3) { sum += points[i]; n++; }
  const avg = n ? sum / n : 0.5;
  return Math.max(0.35, Math.min(1, 0.4 + 0.75 * avg));
}

export const round1 = (v: number) => Math.round(v * 10) / 10;
export const round2 = (v: number) => Math.round(v * 100) / 100;

/**
 * Cleans up raw pen pressure while a stroke is drawn. Pen drivers (Wacom through Windows Ink in
 * particular) often report 0 on pen-down and drop to 0 for single samples mid-stroke; read
 * literally, those become blobs or nicks in the ink. A zero reading repeats the last real one,
 * `null` means no real reading has arrived yet (the caller fills those points in once one does),
 * and light smoothing removes the stair-steps of quantised pressure.
 */
export class PressureFilter {
  private last = -1;
  next(raw: number): number | null {
    if (!(raw > 0)) return this.last < 0 ? null : round2(this.last);
    const v = Math.max(0.05, Math.min(1, raw));
    this.last = this.last < 0 ? v : this.last + (v - this.last) * 0.5;
    return round2(this.last);
  }
}

/** Straight stroke between two points (used for ruler/straight-highlighter/held lines). */
export function straightPoints(ax: number, ay: number, bx: number, by: number, pressure = 0.5): number[] {
  const len = Math.hypot(bx - ax, by - ay);
  const n = Math.max(1, Math.ceil(len / 10));
  const out: number[] = [];
  for (let i = 0; i <= n; i++) out.push(round1(ax + ((bx - ax) * i) / n), round1(ay + ((by - ay) * i) / n), pressure);
  return out;
}
