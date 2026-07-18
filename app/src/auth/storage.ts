import AsyncStorage from '@react-native-async-storage/async-storage';

// Where the key is persisted on this device when the user chooses to save it.
// Web: localStorage. Native: AsyncStorage. Not sent to any server.
// TODO(native): move to expo-secure-store on iOS for at-rest protection.

const STORAGE_KEY = 'sana.key';

export function getStoredKey(): Promise<string | null> {
  return AsyncStorage.getItem(STORAGE_KEY);
}

export function setStoredKey(key: string): Promise<void> {
  return AsyncStorage.setItem(STORAGE_KEY, key);
}

export function clearStoredKey(): Promise<void> {
  return AsyncStorage.removeItem(STORAGE_KEY);
}
