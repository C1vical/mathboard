import { describe, expect, it } from 'vitest';
import { PressureFilter } from '../shared/ink';

describe('PressureFilter', () => {
  it('reports no pressure until the pen sends a real reading', () => {
    const f = new PressureFilter();
    expect(f.next(0)).toBeNull();
    expect(f.next(0)).toBeNull();
    expect(f.next(0.2)).toBe(0.2);
  });

  it('holds the last value through a dropped (zero) reading instead of jumping to half pressure', () => {
    const f = new PressureFilter();
    f.next(0.2);
    expect(f.next(0)).toBe(0.2);
    expect(f.next(NaN)).toBe(0.2);
  });

  it('smooths sudden changes and stays within range', () => {
    const f = new PressureFilter();
    f.next(0.2);
    const v = f.next(1)!;
    expect(v).toBeGreaterThan(0.2);
    expect(v).toBeLessThan(1);
    for (let i = 0; i < 20; i++) f.next(1);
    expect(f.next(1)).toBe(1);
    expect(new PressureFilter().next(0.001)).toBe(0.05);
  });
});
