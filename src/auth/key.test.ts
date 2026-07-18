import { describe, expect, it, vi } from 'vitest';

import { generateKey, isValidKey } from './key';

// expo-crypto needs a native runtime; stand in deterministic bytes that vary per
// call. Randomness quality is expo-crypto's concern — these tests cover encoding
// and validation.
vi.mock('expo-crypto', () => {
  let seed = 0;
  return {
    getRandomBytes: (count: number) => {
      seed += 1;
      return Uint8Array.from({ length: count }, (_, i) => (i * 31 + seed * 131) % 256);
    },
  };
});

describe('generateKey', () => {
  it('produces a 64-char hex key that validates', () => {
    const key = generateKey();
    expect(key).toMatch(/^[0-9a-f]{64}$/);
    expect(isValidKey(key)).toBe(true);
  });

  it('produces distinct keys', () => {
    expect(generateKey()).not.toBe(generateKey());
  });
});

describe('isValidKey', () => {
  it('accepts uppercase hex', () => {
    expect(isValidKey('A'.repeat(64))).toBe(true);
  });

  it('accepts surrounding whitespace', () => {
    expect(isValidKey(`  ${'a'.repeat(64)}\n`)).toBe(true);
  });

  it('rejects empty and wrong-length input', () => {
    expect(isValidKey('')).toBe(false);
    expect(isValidKey('a'.repeat(63))).toBe(false);
    expect(isValidKey('a'.repeat(65))).toBe(false);
  });

  it('rejects non-hex characters', () => {
    expect(isValidKey('g'.repeat(64))).toBe(false);
    expect(isValidKey(`${'a'.repeat(32)} ${'a'.repeat(31)}`)).toBe(false);
  });
});
