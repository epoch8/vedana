import { useEffect, useMemo, useState } from "react";

import Chat from "@/components/blocks/chat/Chat.tsx";
import ChatItem from "@/components/blocks/chat/ChatItem.tsx";

import EntityCardMini from "@/components/blocks/entity-card/EntityCardMini";
import EntityCardMiniList from "@/components/blocks/entity-card/EntityCardMiniList";

import BrandLogo from "@/components/products/vedana/logo/BrandLogo.tsx";

import styles from "./ProductDemo.module.css";

/* =====================================================
   TYPES
===================================================== */

export type ReasoningBlock<CardId extends string> = {
  title?: string;
  narration?: string;
  cards?: readonly CardId[];
  logic?: readonly string[];
};

export type Scenario<CardId extends string> = {
  key: string;
  label: string;
  question: string;
  reasoning: readonly ReasoningBlock<CardId>[];
  answer: string;
};

type Phase =
  | "IDLE"
  | "QUESTION"
  | "THINKING"
  | "REASONING"
  | "ANSWER"
  | "DONE";

type CardBase = {
  entity: string;
  name: string;
  color?: string;
} & Record<string, any>;

type Props<TRegistry extends Record<string, CardBase>> = {
  cardRegistry: TRegistry;
  scenarios: readonly Scenario<keyof TRegistry & string>[];
};

/* =====================================================
   COMPONENT
===================================================== */

export default function ProductDemo<
  TRegistry extends Record<string, CardBase>
>({ cardRegistry, scenarios }: Props<TRegistry>) {

  type CardId = keyof TRegistry & string;
  type Block = ReasoningBlock<CardId>;

  const [phase, setPhase] = useState<Phase>("IDLE");
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [blockIndex, setBlockIndex] = useState<number>(0);

  const [visibleCardCount, setVisibleCardCount] = useState<number>(0);
  const [cardsVisible, setCardsVisible] = useState<boolean>(true);

  const [typedAnswer, setTypedAnswer] = useState<string>("");

  const sleep = (ms: number) =>
    new Promise<void>((resolve) => setTimeout(resolve, ms));

  /* =====================================================
     ACTIVE SCENARIO
  ===================================================== */

  const activeScenario = useMemo(() => {
    return scenarios.find((s) => s.key === activeKey) ?? null;
  }, [scenarios, activeKey]);

  /* =====================================================
     START SCENARIO
  ===================================================== */

  const runScenario = (key: string) => {
    setActiveKey(key);
    setPhase("QUESTION");
    setBlockIndex(0);
    setVisibleCardCount(0);
    setCardsVisible(true);
    setTypedAnswer("");
  };

  const reactions = useMemo(() => {

  if (!activeScenario) return [];

  return scenarios
    .filter((s) => s.key !== activeScenario.key)
    .map((s) => ({
      label: s.label,
      onClick: () => runScenario(s.key)
    }));

}, [scenarios, activeScenario]);

  /* =====================================================
     PHASE MACHINE
  ===================================================== */

  useEffect(() => {
    if (!activeScenario) return;

    if (phase === "QUESTION") {
      const t = setTimeout(() => setPhase("THINKING"), 700);
      return () => clearTimeout(t);
    }

    if (phase === "THINKING") {
      const t = setTimeout(() => {
        setPhase("REASONING");
        setBlockIndex(1);
      }, 1000);
      return () => clearTimeout(t);
    }

  }, [phase, activeScenario]);

  /* =====================================================
     BLOCK FLOW — ANIMATE ONLY CARDS
  ===================================================== */

  useEffect(() => {
    if (!activeScenario) return;
    if (phase !== "REASONING") return;

    const block = activeScenario.reasoning[blockIndex - 1];

    if (!block) {
      const t = setTimeout(() => setPhase("ANSWER"), 600);
      return () => clearTimeout(t);
    }

    let cancelled = false;

    const runBlock = async () => {

      setVisibleCardCount(0);
      setCardsVisible(true);

      if (!block.cards || block.cards.length === 0) {
        await sleep(800);
        if (!cancelled) {
          setBlockIndex((prev) => prev + 1);
        }
        return;
      }

      for (let i = 0; i < block.cards.length; i++) {
        if (cancelled) return;

        // fade out ONLY cards
        setCardsVisible(false);
        await sleep(220);
        if (cancelled) return;

        // update count
        setVisibleCardCount(i + 1);

        // fade in
        setCardsVisible(true);
        await sleep(420); // intentionally slower
      }

      await sleep(600);

      if (!cancelled) {
        setBlockIndex((prev) => prev + 1);
      }
    };

    runBlock();

    return () => {
      cancelled = true;
    };

  }, [blockIndex, phase, activeScenario]);

  /* =====================================================
     ANSWER TYPING
  ===================================================== */

  useEffect(() => {
    if (!activeScenario) return;
    if (phase !== "ANSWER") return;

    let i = 0;
    const text = activeScenario.answer;

    const interval = setInterval(() => {
      setTypedAnswer(text.slice(0, i));
      i++;

      if (i > text.length) {
        clearInterval(interval);
        setPhase("DONE");
      }
    }, 15);

    return () => clearInterval(interval);

  }, [phase, activeScenario]);

  /* =====================================================
     VISIBLE BLOCKS
  ===================================================== */

  const visibleBlocks: readonly Block[] = useMemo(() => {
    if (!activeScenario) return [];
    return activeScenario.reasoning.slice(0, blockIndex);
  }, [activeScenario, blockIndex]);

  /* =====================================================
     HELPERS
  ===================================================== */

 const cardToAttributes = (card: TRegistry[CardId]) => {
  return Object.entries(card)
    .filter(([key]) => !["entity", "name",  "color"].includes(key))
    .map(([key, value]) => {

      // 🔹 LINK ARRAY
      if (Array.isArray(value) && value.length > 0 && typeof value[0] === "object") {
        const rendered = value.map((item: any) => {
          const linked = cardRegistry[item.id as CardId];
          if (!linked) return item.id;

          const attrs = Object.entries(item)
            .filter(([k]) => k !== "id")
            .map(([k, v]) => `${k}: ${v}`)
            .join(", ");

          return attrs
            ? `${linked.name} (${attrs})`
            : linked.name;
        });

        return {
          key,
          value: rendered.join(", ")
        };
      }

      // 🔹 SIMPLE ARRAY
      if (Array.isArray(value)) {
        return {
          key,
          value: value.join(", ")
        };
      }

      // 🔹 PRIMITIVE
      return {
        key,
        value: String(value)
      };
    });
};

  /* =====================================================
     RENDER
  ===================================================== */

  return (
    <div className={styles.wrapper}>
      <div className={styles.layout}>

        {/* ================= CHAT ================= */}
        <div className={styles.chatContainer}>
          <Chat
            title="Chat with Vedana"
            isEmpty={!activeScenario}
            showInput={false}
            emptyScenarios={scenarios.map(s => ({
              key: s.key,
              label: s.label
            }))}
            onScenarioClick={runScenario}
            reactions={reactions}
          >
            {activeScenario && (
              <>
                <ChatItem
                  participant={2}
                  text={activeScenario.question}
                />

                {(phase === "ANSWER" || phase === "DONE") && (
                  <ChatItem participant={1}>
                    <div
                      className={styles.answerContent}
                      dangerouslySetInnerHTML={{ __html: typedAnswer }}
                    />
                  </ChatItem>
                )}
              </>
            )}
          </Chat>
        </div>

        {/* ================= REASONING ================= */}
        <div className={styles.reasoning}>
          {!activeScenario && (
          <div className={styles.reasoningHint}>
            <div className={styles.reasoningHintInner}>

              <div className={styles.reasoningLogo}>
                  <BrandLogo />
                </div>

              <div className={styles.reasoningHintTitle}>
                See Vedana reasoning in action
              </div>

              <div className={styles.reasoningHintText}>
                Ask a question in the chat to generate a reasoning trace.
                <br />
                <br />
                ← start here
                
              </div>

            </div>
          </div>
        )}


          {visibleBlocks.map((block: Block, blockIdx: number) => {

            const isActiveBlock = blockIdx === blockIndex - 1;

            return (
              <div
                key={blockIdx}
                className={`${styles.reasoningBlock} ${
                  isActiveBlock ? styles.reasoningBlockActive : ""
                }`}
              >

                {block.narration && (
                  <div className={styles.reasoningNarration}>
                    {block.narration}
                  </div>
                )}

                {block.title && (
                  <div className={styles.reasoningTitle}>
                    {block.title}
                  </div>
                )}

                {block.cards && block.cards.length > 0 && (
                  <div
                    className={`${styles.cardRailWrapper} ${
                      isActiveBlock && !cardsVisible
                        ? styles.cardRailHidden
                        : ""
                    }`}
                  >
                    <EntityCardMiniList>
                      {(isActiveBlock
                        ? block.cards.slice(0, visibleCardCount)
                        : block.cards
                      ).map((cardId: CardId) => {

                        const card = cardRegistry[cardId];

                        return (
                          <EntityCardMini
                            key={cardId}
                            title={card.name}
                            entityType={card.entity}
                            color={card.color}
                            attributes={cardToAttributes(card)}
                          />
                        );
                      })}
                    </EntityCardMiniList>
                  </div>
                )}

                {block.logic && block.logic.length > 0 && (
                  <div className={styles.logicBlock}>
                    {block.logic.map((line: string, i: number) => (
                      <div key={i} className={styles.logicLine}>
                        {line}
                      </div>
                    ))}
                  </div>
                )}

              </div>
            );
          })}
        </div>

      </div>
    </div>
  );
}