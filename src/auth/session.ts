import { useCallback, useState } from "react";

// In-memory session. The key is held only in state; persisting it to the device
// is a separate explicit action (the "Store on this device" button). No auto-login.
// Logout only ends the session — it does NOT remove a stored key. Forgetting the
// device is the separate "Clear saved key" action.

export function useSession() {
  const [key, setKey] = useState<string | null>(null);

  const login = useCallback((value: string) => {
    setKey(value);
  }, []);

  const logout = useCallback(() => {
    setKey(null);
  }, []);

  return { key, login, logout };
}
