import type { Metadata } from "next";
import Arukikata from "@/components/Arukikata";

export const metadata: Metadata = {
  title: "歩き方 / 設計図 — 変身の縫い目",
  description:
    "この解剖台が何を測り、何を測れなかったかの記録。落ちた判定も、取り下げた主張も、測り違えて直した経緯もそのまま置いてある。",
};

export default function Page() {
  return <Arukikata />;
}
