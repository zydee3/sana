import { Platform } from 'react-native';

// Downloading the key as a file is web-only (uses the browser DOM). On native,
// saving to a file is a separate flow (expo-sharing) that is not built yet.

type WebGlobals = {
  Blob: new (parts: string[], options: { type: string }) => unknown;
  URL: {
    createObjectURL: (blob: unknown) => string;
    revokeObjectURL: (url: string) => void;
  };
  document: {
    createElement: (tag: string) => {
      href: string;
      download: string;
      click: () => void;
    };
  };
};

export function canDownload(): boolean {
  return Platform.OS === 'web';
}

export function downloadKey(key: string): void {
  if (Platform.OS !== 'web') return;
  const web = globalThis as unknown as WebGlobals;
  const blob = new web.Blob([key], { type: 'text/plain' });
  const url = web.URL.createObjectURL(blob);
  const anchor = web.document.createElement('a');
  anchor.href = url;
  anchor.download = 'sana-key.txt';
  anchor.click();
  web.URL.revokeObjectURL(url);
}
