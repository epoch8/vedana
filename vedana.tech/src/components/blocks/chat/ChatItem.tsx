import React from "react";
import styles from "./Chat.module.css";

type Participant = {
  name: string;
};

type Props = {
  participant: 1 | 2;          // 1 = bot, 2 = user (как у тебя)
  text?: string;               // опционально
  children?: React.ReactNode;  // опционально

  participant1?: Participant;  // опционально
  participant2?: Participant;  // опционально
};

const DEFAULTS = {
  bot: { name: "Vedana" },
  user: { name: "You" }
};

export default function ChatItem({
  participant,
  text,
  children,
  participant1,
  participant2
}: Props) {
  const isLeft = participant === 1;

  const p =
    participant1 && participant2
      ? (isLeft ? participant1 : participant2)
      : (isLeft ? DEFAULTS.bot : DEFAULTS.user);

  return (
    <div
      className={`${styles.messageRow} ${
        isLeft ? styles.left : styles.right
      }`}
    >
      {isLeft && (
        <div className={styles.avatar}>
          <div className={styles.avatarPlaceholder} />
        </div>
      )}

      <div className={styles.messageBlock}>
        <div className={styles.name}>{p.name}</div>

        <div className={styles.bubble}>
          {children ?? text}
        </div>
      </div>

      {!isLeft && (
        <div className={styles.avatar}>
          <div className={styles.avatarPlaceholder} />
        </div>
      )}
    </div>
  );
}