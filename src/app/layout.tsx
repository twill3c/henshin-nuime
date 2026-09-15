import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

/** フリート共通フッタの行き先。**推測で書かない** ——
 *  App Menu は app-menu-amber(app-menu.vercel.app は別人のアプリ)。
 *  歩き方と設計図は発行した解説アーティファクトの URL をそのまま使う。 */
const FLEET = {
  license: "https://github.com/twill3c/henshin-nuime/blob/main/LICENSE",
  repository: "https://github.com/twill3c/henshin-nuime",
  arukikata: "https://claude.ai/artifact/EhEzMVV4AGXbp9hF84Ljg8",
  sekkeizu: "https://claude.ai/artifact/UTqki4rHzeUhVFoE6VbG2x",
  appMenu: "https://app-menu-amber.vercel.app/",
} as const;

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
              <Link href="/zure/">ずれの図録</Link>
              <Link href="/nuu/">自分の文を縫う</Link>
              <Link href="/arukikata/">歩き方</Link>
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
        {/* fleet: fixed footer —— フリート共通規約(koho-lens が正本)。
            5 項目・この並び・下部固定。区切りの「・」は文字として置く(CSS で描くと innerText に出ない)。
            © はリンク文言の外、MIT License より後・GitHub より前。
            上の出典表示とは別物なので混ぜない。 */}
        <nav className="fleet" aria-label="フリート共通リンク">
          <a href={FLEET.license} target="_blank" rel="noopener">MIT License</a>
          <span className="fleet__copy"> © 2026 坂田哲朗</span>
          <span className="fsep">・</span>
          <a href={FLEET.repository} target="_blank" rel="noopener">GitHub</a>
          <span className="fsep">・</span>
          <a href={FLEET.arukikata} target="_blank" rel="noopener">変身の縫い目の歩き方</a>
          <span className="fsep">・</span>
          <a href={FLEET.sekkeizu} target="_blank" rel="noopener">変身の縫い目 設計図</a>
          <span className="fsep">・</span>
          <a href={FLEET.appMenu} target="_blank" rel="noopener">App Menu</a>
        </nav>
      </body>
    </html>
  );
}
