export default function Home() {
  return (
    <article style={{ padding: "28px 0 10px", maxWidth: 760 }}>
      <h2 style={{ fontSize: 19, letterSpacing: "0.08em" }}>この解剖台について</h2>
      <p>
        カフカ『変身』は、独語の原文・英訳・日本語訳のすべてが公有で手に入る珍しい作品です。
        同じ物語が三つの言語で並んでいるとき、<strong>どの文がどの文にあたるのか</strong>を、
        機械は正解を教わらずに見つけられるでしょうか。
      </p>
      <p>
        ここでは四つの手がかりで縫い目を引いています ——
        <strong>文字数だけ</strong>(長さモデル)、<strong>位置だけ</strong>(対角線)、
        <strong>意味の近さ</strong>(多言語埋め込み)、そして
        <strong>この一冊だけで学習した Transformer の注意</strong>。
      </p>

      <h3 style={{ fontSize: 16, marginTop: 30 }}>測り方の約束</h3>
      <p>
        縫い目を引く側には、<strong>段落も章も文の番号も渡していません</strong>。
        渡してしまえば「段落を守る縫い目が引けた」は当たり前になるからです。
        段落は、引き終わったあとの<strong>採点にだけ</strong>使います。
      </p>
      <p>
        だから「独語の第 k 段落の文が、英語の第 k 段落の文に結ばれたか」という問いは、
        機械が一度も見ていない答え合わせになります。
      </p>

      <h3 style={{ fontSize: 16, marginTop: 30 }}>分かったこと</h3>
      <ul>
        <li>
          <strong>独語と英語は、文字数だけでほぼ縫える。</strong>
          段落一致率 0.9949。位置だけの対角線は 0.5577 なので、これは差として読めます。
        </li>
        <li>
          <strong>日本語との間では、文字数は手がかりにならない。</strong>
          対角線とほとんど区別がつきません。表記体系をまたぐと文字数は効かないのです。
        </li>
        <li>
          <strong>そこで効くのが意味の近さ。</strong>
          埋め込みなら独語↔日本語も縫えます(細分の破れ 1 対 56)。
        </li>
        <li>
          <strong>一冊だけで学習した Transformer は、対応を学ばなかった。</strong>
          これは事前に登録した予測で、<strong>外れた側</strong>です。
          翻訳ができないことは予測どおりでしたが、注意が対応を捉えるという後半は
          統計的に確かめられませんでした(p = 0.105)。予測は消さずに残してあります。
        </li>
      </ul>

      <p className="note" style={{ marginTop: 26 }}>
        設計・品質ゲート・測って捨てた記録は、リポジトリの SPEC.md にあります。
      </p>
    </article>
  );
}
