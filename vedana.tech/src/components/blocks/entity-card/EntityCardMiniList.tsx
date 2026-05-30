import React from "react";
import styles from "./EntityCard.module.css";

type Props = {
  children: React.ReactNode;
  className?: string;
};

export default function EntityCardMiniList({
  children,
  className
}: Props) {
  return (
    <div
      className={[
        styles["entity-card-mini-rail"],
        className
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </div>
  );
}