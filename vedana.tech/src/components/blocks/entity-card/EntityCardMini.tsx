// EntityCardMini.tsx

import styles from "./EntityCard.module.css";

export type Attribute = {
  key: string;
  value?: string;
};

type Props = {
  title: string;
  entityType: string;
  color?: string;
  attributes: Attribute[];
  className?: string;
};

export default function EntityCardMini({
  title,
  entityType,
  color,
  attributes,
  className
}: Props) {
  return (
    <article
      className={`${styles["entity-card"]} ${styles["entity-card-mini"]} ${className ?? ""}`}
      data-entity-type={entityType}
      style={color ? { ["--entity-accent" as any]: color } : undefined}
    >
      <header className={styles["entity-card__header"]}>
        <div className={styles["entity-card__title"]}>
          {title}
        </div>
      </header>

      <div className={styles["entity-card__body"]}>
        {attributes.map((attr, index) => (
          <div key={index} className={styles["entity-item"]}>
            <div className={styles["entity-item__key"]}>
              {attr.key}
            </div>

            <div className={styles["entity-item__value"]}>
              {attr.value}
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}