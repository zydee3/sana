import AsyncStorage from "@react-native-async-storage/async-storage";

import type { Chat } from "./types";

// Chats persist locally on the device. Not encrypted or synced yet — that comes
// with the backend (encrypt-at-rest with the user's key, decrypt server-side for claude -p).

const CHATS_KEY = "sana.chats";

export async function loadChats(): Promise<Chat[]> {
  const raw = await AsyncStorage.getItem(CHATS_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as Chat[];
  } catch {
    return [];
  }
}

export async function saveChats(chats: Chat[]): Promise<void> {
  await AsyncStorage.setItem(CHATS_KEY, JSON.stringify(chats));
}
