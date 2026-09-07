import type { Metadata } from "next";
import Reader from "@/components/Reader";

export const metadata: Metadata = {
  title: "三面の本文 — 変身の縫い目",
  description:
    "独語原文・英訳・日本語訳を三列に並べ、文をひとつ選ぶと対応する文が光る。手法を切り替えて、縫い目の違いを見る。",
};

export default function Page() {
  return <Reader />;
}
