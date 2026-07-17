import { useCallback, useState } from "react";

import { clearStoredKey } from "./storage";

// In-memory session. The key is held only in state; persisting it to the device
// is a separate explicit action (the "Store on this device" button). No auto-login.

export function useSession() {
  const [key, setKey] = useState<string | null>(null);

  const login = useCallback((value: string) => {
    setKey(value);
  }, []);

  const logout = useCallback(async () => {
    await clearStoredKey();
    setKey(null);
  }, []);

  return { key, login, logout };
}
