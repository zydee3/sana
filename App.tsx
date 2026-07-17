import { useEffect, useState, type ReactNode } from "react";
import { Pressable, View } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import * as Clipboard from "expo-clipboard";
import { Check, Clipboard as ClipboardIcon } from "lucide-react-native";

import "./global.css";
import { Button } from "@/components/ui/button";
import { Text } from "@/components/ui/text";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Icon } from "@/components/ui/icon";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
    return <Register onBack={() => setView("landing")} onLogin={() => setView("login")} />;
  }
  if (view === "login") {
    return <Login onDone={onAuthenticated} onBack={() => setView("landing")} />;
  }
  return (
    <Screen>
      <View className="w-full max-w-sm items-center gap-10">
        <Text className="text-foreground text-5xl font-bold tracking-tight">
          Sana
        </Text>
        <View className="w-full gap-3">
          <Button onPress={() => setView("register")}>
            <Text>Register</Text>
          </Button>
          <Button variant="outline" onPress={() => setView("login")}>
            <Text>Login</Text>
          </Button>
        </View>
      </View>
    </Screen>
  );
}

function Register({
  onBack,
  onLogin,
}: {
  onBack: () => void;
  onLogin: () => void;
}) {
  const [key] = useState(generateKey);
  const [copied, setCopied] = useState(false);
  const [stored, setStored] = useState(false);

  async function copy() {
    await Clipboard.setStringAsync(key);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function store() {
    await setStoredKey(key);
    setStored(true);
  }

  return (
    <Screen>
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Your key</CardTitle>
          <CardDescription>
            This key is your only way to log in. It is never stored on our
            servers — save it somewhere safe.
          </CardDescription>
        </CardHeader>
        <CardContent className="gap-4">
          <View className="border-border bg-muted relative rounded-md border p-3 pr-11">
            <Text selectable className="font-mono text-xs">
              {key}
            </Text>
            <Pressable
              onPress={copy}
              hitSlop={8}
              accessibilityLabel="Copy key"
              className="active:bg-background absolute right-1 top-1 rounded-md p-2"
            >
              <Icon
                as={copied ? Check : ClipboardIcon}
                size={18}
                className="text-muted-foreground"
              />
            </Pressable>
          </View>
          <View className="gap-2">
            <Button variant="outline" onPress={store}>
              <Text>{stored ? "Stored on this device" : "Store on this device"}</Text>
            </Button>
            {canDownload() && (
              <Button variant="outline" onPress={() => downloadKey(key)}>
                <Text>Download key</Text>
              </Button>
            )}
          </View>
        </CardContent>
        <CardFooter className="gap-3">
          <Button className="flex-1" variant="ghost" onPress={onBack}>
            <Text>Back</Text>
          </Button>
          <Button className="flex-1" onPress={onLogin}>
            <Text>Login</Text>
          </Button>
        </CardFooter>
      </Card>
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
  const [entry, setEntry] = useState("");
  const [savedKey, setSavedKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStoredKey().then(setSavedKey);
  }, []);

  function submitEntry() {
    const trimmed = entry.trim();
    if (!isValidKey(trimmed)) {
      setError("That does not look like a valid key.");
      return;
    }
    onDone(trimmed);
  }

  return (
    <Screen>
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Log in</CardTitle>
        </CardHeader>
        <CardContent className="gap-4">
          <View className="gap-1.5">
            <Label>Your key</Label>
            <Input
              value={entry}
              onChangeText={(text) => {
                setEntry(text);
                setError(null);
              }}
              placeholder="Paste your key"
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>
          {error && <Text className="text-destructive text-sm">{error}</Text>}
          <Button onPress={submitEntry}>
            <Text>Log in</Text>
          </Button>
          {savedKey && isValidKey(savedKey) && (
            <Button variant="outline" onPress={() => onDone(savedKey)}>
              <Text>Use saved key</Text>
            </Button>
          )}
        </CardContent>
        <CardFooter>
          <Button className="flex-1" variant="ghost" onPress={onBack}>
            <Text>Back</Text>
          </Button>
        </CardFooter>
      </Card>
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
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Sana</CardTitle>
          <CardDescription>You are logged in.</CardDescription>
        </CardHeader>
        <CardContent className="gap-3">
          {canDownload() && (
            <Button variant="outline" onPress={onDownload}>
              <Text>Download key</Text>
            </Button>
          )}
          <Button variant="secondary" onPress={onLogout}>
            <Text>Log out</Text>
          </Button>
        </CardContent>
      </Card>
    </Screen>
  );
}

function Screen({ children }: { children: ReactNode }) {
  return (
    <View className="bg-background flex-1 items-center justify-center px-6">
      {children}
    </View>
  );
}
