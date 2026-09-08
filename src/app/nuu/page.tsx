import type { Metadata } from "next";
import Nuu from "@/components/Nuu";

export const metadata: Metadata = {
  title: "自分の文を縫う — 変身の縫い目",
  description:
    "文をひとつ打つと、ブラウザの中で埋め込み、三つの版のどこに近いかを探す。文はどこにも送られない。",
};

export default function Page() {
  return <Nuu />;
}
