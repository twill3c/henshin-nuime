import type { Metadata } from "next";
import Kotoba from "@/components/Kotoba";

export const metadata: Metadata = {
  title: "一語の変身 — 変身の縫い目",
  description:
    "独語の一語が、英訳・原田訳・自前訳へどう移ったか。訳語は登録簿で一つに固定してあり、選んだ理由をそのまま出す。",
};

export default function Page() {
  return <Kotoba />;
}
