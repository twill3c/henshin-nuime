import type { Metadata } from "next";
import AttentionBand from "@/components/AttentionBand";

export const metadata: Metadata = {
  title: "注意の帯 — 変身の縫い目",
  description:
    "『変身』一冊だけで学習した Transformer の cross-attention を帯として見る。事前に登録した予測と、その落ちた判定。",
};

export default function Page() {
  return <AttentionBand />;
}
