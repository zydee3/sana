import { Pressable, ScrollView, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Download, LogOut, Plus } from "lucide-react-native";

import { Text } from "@/components/ui/text";
import { Button } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { canDownload, downloadKey } from "@/auth/download";
import type { Chat } from "./types";

export function ChatListScreen({
  chats,
  loaded,
  onNewChat,
  onOpenChat,
  sessionKey,
  onLogout,
}: {
  chats: Chat[];
  loaded: boolean;
  onNewChat: () => void;
  onOpenChat: (chatId: string) => void;
  sessionKey: string;
  onLogout: () => void;
}) {
  return (
    <SafeAreaView className="bg-background flex-1">
      <View className="flex-row items-center justify-between px-4 py-2">
        <Text className="text-2xl font-bold">Sana</Text>
        <View className="flex-row">
          {canDownload() && (
            <Button
              variant="ghost"
              size="icon"
              onPress={() => downloadKey(sessionKey)}
              accessibilityLabel="Download key"
            >
              <Icon as={Download} size={20} />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            onPress={onLogout}
            accessibilityLabel="Log out"
          >
            <Icon as={LogOut} size={20} />
          </Button>
        </View>
      </View>

      <View className="px-4 pb-3">
        <Button onPress={onNewChat}>
          <Icon as={Plus} size={18} />
          <Text>New chat</Text>
        </Button>
      </View>

      <ScrollView className="flex-1 px-4">
        <View className="gap-2 pb-8">
          {loaded && chats.length === 0 && (
            <Text className="text-muted-foreground mt-10 text-center">
              No chats yet. Start one above.
            </Text>
          )}
          {chats.map((chat) => (
            <Pressable
              key={chat.id}
              onPress={() => onOpenChat(chat.id)}
              className="border-border bg-card active:bg-muted rounded-lg border p-4"
            >
              <Text className="font-medium" numberOfLines={1}>
                {chat.title}
              </Text>
              <Text className="text-muted-foreground mt-1 text-sm" numberOfLines={1}>
                {lastPreview(chat)}
              </Text>
            </Pressable>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function lastPreview(chat: Chat): string {
  const last = chat.messages[chat.messages.length - 1];
  return last ? last.content : "No messages yet";
}
