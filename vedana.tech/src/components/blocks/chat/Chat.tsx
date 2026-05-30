import React from "react";
import styles from "./Chat.module.css";

export type EmptyScenario = {
  key: string;
  label: string;
};

export type ChatReaction = {
  label: string;
  onClick: () => void;
};

export type ChatProps = {
  title?: string;

  showDate?: boolean;
  dateLabel?: string;

  reactions?: ChatReaction[];

  inputPlaceholder?: string;
  inputValue?: string;

  showSendButton?: boolean;
  showInput?: boolean;

  isEmpty?: boolean;

  emptyScenarios?: EmptyScenario[];
  onScenarioClick?: (key: string) => void;

  children?: React.ReactNode;
};

export default function Chat({
  title,
  showDate = false,
  dateLabel,
  reactions = [],
  inputPlaceholder = "Type a message",
  inputValue = "",
  showSendButton = true,
  showInput = true,
  isEmpty = false,

  emptyScenarios = [] as EmptyScenario[],

  onScenarioClick,
  children
}: ChatProps) {
  return (
    <div className={styles.chat}>
      {title && (
        <div className={styles.header}>
          <h3 className={styles.title}>{title}</h3>
        </div>
      )}

      {/* ==============================
          MESSAGES
      ============================== */}

      <div className={styles.messages}>
        {isEmpty ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyInner}>
              <h4 className={styles.emptyTitle}>
                Ask Vedana a real business question
              </h4>

              <p className={styles.emptySubtitle}>
                Answers are generated from verified graph logic.
              </p>

              <div className={styles.emptyActions}>
                {emptyScenarios.map((s, i) => (
                  <button
                    key={s.key ?? i}
                    type="button"
                    className={styles.scenarioCard}
                    onClick={() => onScenarioClick?.(s.key)}
                  >
                    <span className={styles.scenarioLabel}>
                      {s.label}
                    </span>
                    <span className={styles.scenarioHint}>
                      Try this example
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            {showDate && dateLabel && (
              <div className={styles.date}>{dateLabel}</div>
            )}

            {children}
          </>
        )}
      </div>

      {/* ==============================
          REACTIONS (always visible)
      ============================== */}

      {reactions.length > 0 && (
        <div className={styles.reactions}>
          {reactions.map((r, i) => (
            <button
              key={i}
              type="button"
              className={styles.reaction}
              onClick={r.onClick}
            >
              {r.label}
            </button>
          ))}
        </div>
      )}

      {/* ==============================
          INPUT AREA (optional)
      ============================== */}

      {showInput && (
        <div className={styles.inputArea}>
          <div className={styles.inputWrapper}>
            {inputValue ? (
              <span className={styles.inputValue}>{inputValue}</span>
            ) : (
              <span className={styles.placeholder}>
                {isEmpty
                  ? "Or type your own question..."
                  : inputPlaceholder}
              </span>
            )}
          </div>

          {showSendButton && (
            <button type="button" className={styles.sendBtn}>
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className={styles.sendIcon}
              >
                <path d="M3.714 3.048a.498.498 0 0 0-.683.627l2.843 7.627a2 2 0 0 1 0 1.396l-2.842 7.627a.498.498 0 0 0 .682.627l18-8.5a.5.5 0 0 0 0-.904z" />
                <path d="M6 12h16" />
              </svg>
            </button>
          )}
        </div>
      )}
    </div>
  );
}