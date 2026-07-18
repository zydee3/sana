import { useEffect, useState, type ReactNode } from "react";
import { Modal, Pressable, View } from "react-native";
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
import { generateKey, isValidKey } from "@/auth/key";
import { useSession } from "@/auth/session";
import { clearStoredKey, getStoredKey, setStoredKey } from "@/auth/storage";
import { canDownload, downloadKey } from "@/auth/download";
import { Main } from "@/chat/Main";

export default function App() {
  const { key, login, logout } = useSession();

  return (
    <SafeAreaProvider>
      {key ? (
        <Main sessionKey={key} onLogout={logout} />
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
      <View className="w-full max-w-sm items-center gap-8">
        <Text className="text-foreground text-5xl font-bold tracking-tight">
          Register
        </Text>
        <View className="w-full gap-4">
          <Text className="text-muted-foreground text-center">
            This key is your only way to log in. It is never stored on our
            servers — save it somewhere safe.
          </Text>
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
          <View className="flex-row gap-2">
            <Button className="flex-1" variant="ghost" onPress={onBack}>
              <Text>Back</Text>
            </Button>
            <Button className="flex-1" onPress={onLogin}>
              <Text>Login</Text>
            </Button>
          </View>
        </View>
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
  const [entry, setEntry] = useState("");
  const [savedKey, setSavedKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showClear, setShowClear] = useState(false);

  useEffect(() => {
    getStoredKey().then(setSavedKey);
  }, []);

  const hasSavedKey = savedKey !== null && isValidKey(savedKey);

  function submitEntry() {
    const trimmed = entry.trim();
    if (!isValidKey(trimmed)) {
      setError("That does not look like a valid key.");
      return;
    }
    onDone(trimmed);
  }

  async function clearKey() {
    await clearStoredKey();
    setSavedKey(null);
    setShowClear(false);
  }

  return (
    <Screen>
      <View className="w-full max-w-sm items-center gap-8">
        <Text className="text-foreground text-5xl font-bold tracking-tight">
          Login
        </Text>
        <View className="w-full gap-4">
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
              onSubmitEditing={submitEntry}
              returnKeyType="send"
            />
          </View>
          {error && <Text className="text-destructive text-sm">{error}</Text>}
          <Button onPress={submitEntry}>
            <Text>Log in</Text>
          </Button>
          <SavedKeyButton
            hasKey={hasSavedKey}
            onUse={() => {
              if (savedKey && isValidKey(savedKey)) onDone(savedKey);
            }}
          />
          <View className="flex-row gap-2">
            <Button className="flex-1" variant="ghost" onPress={onBack}>
              <Text>Back</Text>
            </Button>
            {hasSavedKey && (
              <Button
                className="flex-1"
                variant="ghost"
                onPress={() => setShowClear(true)}
              >
                <Text className="text-destructive">Clear saved key</Text>
              </Button>
            )}
          </View>
        </View>
      </View>
      <ClearKeyDialog
        visible={showClear}
        keyValue={savedKey}
        onClear={clearKey}
        onClose={() => setShowClear(false)}
      />
    </Screen>
  );
}

function SavedKeyButton({
  hasKey,
  onUse,
}: {
  hasKey: boolean;
  onUse: () => void;
}) {
  const [showHint, setShowHint] = useState(false);

  if (hasKey) {
    return (
      <Button variant="outline" onPress={onUse}>
        <Text>Use saved key</Text>
      </Button>
    );
  }

  return (
    <View className="relative">
      {showHint && (
        <View className="absolute bottom-full left-0 right-0 mb-2 items-center">
          <View className="bg-foreground max-w-[15rem] rounded-md px-3 py-2">
            <Text className="text-background text-center text-sm">
              No key is saved on this device.
            </Text>
          </View>
          <View className="border-x-[6px] border-t-[6px] border-x-transparent border-t-foreground h-0 w-0" />
        </View>
      )}
      <Button
        variant="outline"
        className="opacity-50"
        onHoverIn={() => setShowHint(true)}
        onHoverOut={() => setShowHint(false)}
        onPress={() => setShowHint(true)}
      >
        <Text>Use saved key</Text>
      </Button>
    </View>
  );
}

function ClearKeyDialog({
  visible,
  keyValue,
  onClear,
  onClose,
}: {
  visible: boolean;
  keyValue: string | null;
  onClear: () => void;
  onClose: () => void;
}) {
  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <Pressable
        className="flex-1 items-center justify-center bg-black/50 px-6"
        onPress={onClose}
      >
        <Pressable
          className="border-border bg-card w-full max-w-sm gap-3 rounded-lg border p-5"
          onPress={() => {}}
        >
          <Text className="text-lg font-semibold">Clear saved key?</Text>
          <Text className="text-muted-foreground">
            This removes the key saved on this device. You'll need it to log in
            again, so download it first if you haven't saved it elsewhere.
          </Text>
          {canDownload() && keyValue && (
            <Button variant="outline" onPress={() => downloadKey(keyValue)}>
              <Text>Download key</Text>
            </Button>
          )}
          <View className="flex-row gap-2">
            <Button className="flex-1" variant="ghost" onPress={onClose}>
              <Text>Cancel</Text>
            </Button>
            <Button className="flex-1" variant="destructive" onPress={onClear}>
              <Text>Clear</Text>
            </Button>
          </View>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

function Screen({ children }: { children: ReactNode }) {
  return (
    <View className="bg-background flex-1 items-center justify-center px-6">
      {children}
    </View>
  );
}
