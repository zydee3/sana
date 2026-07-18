import { useState } from "react";

import { useChats } from "./useChats";
import { ChatListScreen } from "./ChatListScreen";
import { ChatScreen } from "./ChatScreen";

export function Main({
  sessionKey,
  onLogout,
}: {
  sessionKey: string;
  onLogout: () => void;
}) {
  const { chats, loaded, createChat, sendMessage } = useChats();
  const [openChatId, setOpenChatId] = useState<string | null>(null);

  const openChat = openChatId ? (chats.find((c) => c.id === openChatId) ?? null) : null;

  if (openChat) {
    return (
      <ChatScreen
        chat={openChat}
        onBack={() => setOpenChatId(null)}
        onSend={(content) => sendMessage(openChat.id, content)}
      />
    );
  }

  return (
    <ChatListScreen
      chats={chats}
      loaded={loaded}
      onNewChat={() => setOpenChatId(createChat().id)}
      onOpenChat={setOpenChatId}
      sessionKey={sessionKey}
      onLogout={onLogout}
    />
  );
}
