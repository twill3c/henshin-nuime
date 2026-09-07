import type { Metadata } from "next";
import SeamMap from "@/components/SeamMap";

export const metadata: Metadata = {
  title: "縫い目の地図 — 変身の縫い目",
  description:
    "三つの手法が引いた縫い目の経路を同じ軸に重ね、段落の切れ目を薄い格子として敷く。オラクルを目に見える形にした図。",
};

export default function Page() {
  return <SeamMap />;
}
