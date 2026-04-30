import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Ant-RH Dashboard (Next)",
  description: "Local Ant-RH monitoring and control dashboard"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="appShell">
          <header className="topbar">
            <div className="brand">
              <div className="logo">Ant-RH</div>
              <div className="subtitle">Next.js Dashboard</div>
            </div>
            <div className="topbarRight">
              <div className="envTag">
                API: <span className="mono">{process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8084"}</span>
              </div>
            </div>
          </header>
          <main className="container">{children}</main>
          <footer className="footer mono">Local only · Next.js frontend · FastAPI backend</footer>
        </div>
      </body>
    </html>
  );
}

