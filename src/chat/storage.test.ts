import { beforeEach, describe, expect, it, vi } from 'vitest';

import { loadChats, saveChats } from './storage';

import type { Chat } from './types';

// AsyncStorage needs a native runtime; back it with an in-memory map.
const store = vi.hoisted(() => new Map<string, string>());
vi.mock('@react-native-async-storage/async-storage', () => ({
  default: {
    getItem: async (key: string) => store.get(key) ?? null,
    setItem: async (key: string, value: string) => {
      store.set(key, value);
    },
  },
}));

const chat: Chat = {
  id: 'c1',
  title: 'hello',
  messages: [{ id: 'm1', role: 'user', content: 'hi', createdAt: 1 }],
  createdAt: 1,
  updatedAt: 2,
};

beforeEach(() => {
  store.clear();
});

describe('chat storage', () => {
  it('returns [] when nothing is stored', async () => {
    expect(await loadChats()).toEqual([]);
  });

  it('round-trips chats through save and load', async () => {
    await saveChats([chat]);
    expect(await loadChats()).toEqual([chat]);
  });

  it('returns [] on corrupted stored JSON', async () => {
    store.set('sana.chats', '{not json');
    expect(await loadChats()).toEqual([]);
  });
});
