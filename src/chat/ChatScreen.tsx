import { useRef, useState } from "react";
import { KeyboardAvoidingView, Platform, ScrollView, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { ArrowLeft, Send } from "lucide-react-native";

import { Text } from "@/components/ui/text";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Icon } from "@/components/ui/icon";
import type { Chat } from "./types";

export function ChatScreen({
  chat,
  onBack,
  onSend,
}: {
  chat: Chat;
  onBack: () => void;
  onSend: (content: string) => void;
}) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<ScrollView>(null);

  function send() {
    const text = draft.trim();
    if (!text) return;
    onSend(text);
    setDraft("");
  }

  return (
    <SafeAreaView className="bg-background flex-1">
      <View className="border-border flex-row items-center gap-1 border-b px-2 py-2">
        <Button variant="ghost" size="icon" onPress={onBack} accessibilityLabel="Back">
          <Icon as={ArrowLeft} size={22} />
        </Button>
        <Text className="flex-1 font-medium" numberOfLines={1}>
          {chat.title}
        </Text>
      </View>

      <KeyboardAvoidingView
        className="flex-1"
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <ScrollView
          ref={scrollRef}
          className="flex-1"
          onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
        >
          <View className="gap-3 p-4">
            {chat.messages.length === 0 && (
              <Text className="text-muted-foreground mt-10 text-center">
                How are you feeling today?
              </Text>
            )}
            {chat.messages.map((message) => (
              <View
                key={message.id}
                className={message.role === "user" ? "items-end" : "items-start"}
              >
                <View
                  className={`max-w-[85%] rounded-2xl px-4 py-2 ${
                    message.role === "user" ? "bg-primary" : "bg-muted"
                  }`}
                >
                  <Text
                    className={
                      message.role === "user" ? "text-primary-foreground" : "text-foreground"
                    }
                  >
                    {message.content}
                  </Text>
                </View>
              </View>
            ))}
          </View>
        </ScrollView>

        <View className="border-border flex-row items-center gap-2 border-t p-3">
          <Input
            className="flex-1"
            value={draft}
            onChangeText={setDraft}
            placeholder="Message Sana"
            onSubmitEditing={send}
            returnKeyType="send"
          />
          <Button size="icon" onPress={send} accessibilityLabel="Send">
            <Icon as={Send} size={18} />
          </Button>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
