import * as Crypto from 'expo-crypto';

// The login credential is 256 bits of random material, encoded as hex.
// Generated client-side, never sent to a server. Hex (not base64) keeps
// encoding trivial and identical on web and native.

const KEY_BYTES = 32;

export function generateKey(): string {
  const bytes = Crypto.getRandomBytes(KEY_BYTES);
  return bytesToHex(bytes);
}

export function isValidKey(value: string): boolean {
  return /^[0-9a-f]{64}$/i.test(value.trim());
}

function bytesToHex(bytes: Uint8Array): string {
  let hex = '';
  for (const byte of bytes) hex += byte.toString(16).padStart(2, '0');
  return hex;
}
