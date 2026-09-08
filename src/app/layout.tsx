import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "変身の縫い目",
  description:
    "カフカ『変身』を独語原文・英訳・日本語訳の三面で並べ、どの文がどの文に対応するかを教師なしで引かせ、その縫い目を測る解剖台。",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body>
        <header className="site-head">
          <div className="wrap">
            <h1>変身の縫い目</h1>
            <p>
              カフカ『変身』を独語原文・英訳・日本語訳の三面で並べ、
              どの文がどの文に対応するかを<strong>教師なしで</strong>引かせ、その縫い目を測る解剖台。
            </p>
            <nav className="nav">
              <Link href="/">はじめに</Link>
              <Link href="/yomu/">三面の本文</Link>
              <Link href="/chizu/">縫い目の地図</Link>
              <Link href="/obi/">注意の帯</Link>
              <Link href="/kotoba/">一語の変身</Link>
              <Link href="/nuu/">自分の文を縫う</Link>
            </nav>
          </div>
        </header>
        <main className="wrap">{children}</main>
        <footer className="site-foot">
          <div className="wrap">
            <ul>
              <li>
                本文は三点とも公有。Die Verwandlung(
                <a href="https://www.gutenberg.org/ebooks/22367">PG #22367</a>)/
                Metamorphosis, translated by David Wyllie(
                <a href="https://www.gutenberg.org/ebooks/5200">PG #5200</a>)/
                変身 原田義人訳(
                <a href="https://www.aozora.gr.jp/cards/001235/card49866.html">
                  青空文庫 49866
                </a>
                )
              </li>
              <li>
                日本語版の底本は「世界文学大系58　カフカ」筑摩書房、1960（昭和35）年4月10日発行。
                入力: kompass / 校正: 青空文庫。2010年11月28日作成 / 2016年2月22日修正。
                本サイトはこのファイルのルビ(振仮名)を本文から外している。
              </li>
              <li>
                「青空文庫」「Project Gutenberg」の名称は出所表示として用いており、
                各団体の関与・認知・許可を示すものではない。
              </li>
              <li>非商用の実装訓練として作っている。cron・DB・サーバ関数を持たない。</li>
            </ul>
          </div>
        </footer>
      </body>
    </html>
  );
}
