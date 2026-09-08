import type { Metadata } from "next";
import Zure from "@/components/Zure";

export const metadata: Metadata = {
  title: "ずれの図録 — 変身の縫い目",
  description:
    "分割と併合の形、段落を守れたか、三角整合、そして訳者差。どれも別のものを見ているので、一つの数字にまとめずに並べる。",
};

export default function Page() {
  return <Zure />;
}
