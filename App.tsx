import { useEffect, useState, type ReactNode } from "react";
import { View } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import * as Clipboard from "expo-clipboard";

import "./global.css";
import { Button } from "@/components/ui/button";
import { Text } from "@/components/ui/text";
import { generateKey, isValidKey } from "@/auth/key";
import { useSession } from "@/auth/session";
import { getStoredKey, setStoredKey } from "@/auth/storage";
import { canDownload, downloadKey } from "@/auth/download";

export default function App() {
  const { key, login, logout } = useSession();

  return (
    <SafeAreaProvider>
      {key ? (
        <Home onDownload={() => downloadKey(key)} onLogout={logout} />
      ) : (
        <Auth onAuthenticated={login} />
      )}
      <StatusBar style="auto" />
    </SafeAreaProvider>
  );
}

type AuthView = "landing" | "register" | "login";

function Auth({ onAuthenticated }: { onAuthenticated: (key: string) => void }) {
  const [view, setView] = useState<AuthView>("landing");

  if (view === "register") {
    return <Register onDone={onAuthenticated} onBack={() => setView("landing")} />;
  }
  if (view === "login") {
    return <Login onDone={onAuthenticated} onBack={() => setView("landing")} />;
  }
  return (
    <Screen>
      <Text variant="h1">Sana</Text>
      <View className="w-64 gap-3">
        <Button onPress={() => setView("register")}>
          <Text>Register</Text>
        </Button>
        <Button variant="outline" onPress={() => setView("login")}>
          <Text>Login</Text>
        </Button>
      </View>
    </Screen>
  );
}

function Register({
  onDone,
  onBack,
}: {
  onDone: (key: string) => void;
  onBack: () => void;
}) {
  const [key] = useState(generateKey);
  const [copied, setCopied] = useState(false);
  const [stored, setStored] = useState(false);

  async function copy() {
    await Clipboard.setStringAsync(key);
    setCopied(true);
  }

  async function store() {
    await setStoredKey(key);
    setStored(true);
  }

  return (
    <Screen>
      <Text variant="h2">Your key</Text>
      <Text variant="muted" className="max-w-xs text-center">
        This key is your only way to log in. It is never stored on our servers.
        Save it somewhere safe.
      </Text>
      <View className="w-64 rounded-lg border border-border bg-card p-3">
        <Text selectable className="font-mono text-xs">
          {key}
        </Text>
      </View>
      <View className="w-64 gap-3">
        <Button variant="outline" onPress={copy}>
          <Text>{copied ? "Copied" : "Copy key"}</Text>
        </Button>
        <Button variant="outline" onPress={store}>
          <Text>{stored ? "Stored on this device" : "Store on this device"}</Text>
        </Button>
        {canDownload() && (
          <Button variant="outline" onPress={() => downloadKey(key)}>
            <Text>Download key</Text>
          </Button>
        )}
        <Button onPress={() => onDone(key)}>
          <Text>Continue</Text>
        </Button>
        <Button variant="link" onPress={onBack}>
          <Text>Back</Text>
        </Button>
      </View>
    </Screen>
  );
}

function Login({
  onDone,
  onBack,
}: {
  onDone: (key: string) => void;
  onBack: () => void;
}) {
  const [savedKey, setSavedKey] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStoredKey().then((stored) => {
      setSavedKey(stored);
      setChecked(true);
    });
  }, []);

  const hasSavedKey = savedKey !== null && isValidKey(savedKey);

  function loginFromDevice() {
    if (savedKey && isValidKey(savedKey)) {
      onDone(savedKey);
    } else {
      setError("No saved key on this device.");
    }
  }

  return (
    <Screen>
      <Text variant="h2">Log in</Text>
      {error && <Text className="text-destructive">{error}</Text>}
      <View className="w-64 gap-3">
        <Button onPress={loginFromDevice} disabled={!hasSavedKey}>
          <Text>Log in with saved key</Text>
        </Button>
        <Button variant="link" onPress={onBack}>
          <Text>Back</Text>
        </Button>
      </View>
      {checked && !hasSavedKey && (
        <Text variant="muted" className="max-w-xs text-center text-sm">
          No saved key on this device. Register to create one. Logging in from a
          key file is coming next.
        </Text>
      )}
    </Screen>
  );
}

function Home({
  onDownload,
  onLogout,
}: {
  onDownload: () => void;
  onLogout: () => void;
}) {
  return (
    <Screen>
      <Text variant="h1">Sana</Text>
      <Text variant="muted">You are logged in.</Text>
      <View className="w-64 gap-3">
        {canDownload() && (
          <Button variant="outline" onPress={onDownload}>
            <Text>Download key</Text>
          </Button>
        )}
        <Button onPress={onLogout}>
          <Text>Log out</Text>
        </Button>
      </View>
    </Screen>
  );
}

function Screen({ children }: { children: ReactNode }) {
  return (
    <View className="flex-1 items-center justify-center gap-6 bg-background px-6">
      {children}
    </View>
  );
}
