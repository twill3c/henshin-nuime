import type { NextConfig } from "next";

// 静的書き出しのみ。**サーバ関数を一つも持たない**(SPEC N-01 / G-11)。
// 縫い目も本文も pipeline/bake.py がビルド前に public/data へ焼き込むので、
// ブラウザは静的ファイルを読むだけでよい。cron も DB も無い。
const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
