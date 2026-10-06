import { useEffect, useState } from "react";
import { Console } from "./console.tsx";
import { Docs } from "./docs.tsx";
import { Marketing } from "./marketing.tsx";
import { surfaceForHost } from "./security.ts";

type Surface = "marketing" | "console" | "docs";

function initialSurface(): Surface {
  const hostSurface = surfaceForHost(location.hostname);
  if (hostSurface === "console" || hostSurface === "docs") return hostSurface;
  if (location.pathname.startsWith("/console")) return "console";
  if (location.pathname.startsWith("/docs")) return "docs";
  return "marketing";
}

export function App() {
  const [surface, setSurface] = useState<Surface>(initialSurface);
  const [theme, setTheme] = useState("light");
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  function navigate(path: string) {
    const next: Surface = path.startsWith("/console") ? "console" : path.startsWith("/docs") ? "docs" : "marketing";
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
      history.pushState({}, "", next === "marketing" ? "/" : `/${next}`);
    }
    setSurface(next);
  }
  return (
    <>
      <a className="skip" href="#top">Skip to content</a>
      {surface === "console" ? <Console onNavigate={navigate} theme={theme} setTheme={setTheme} /> : null}
      {surface === "docs" ? <Docs onNavigate={navigate} /> : null}
      {surface === "marketing" ? <Marketing onNavigate={navigate} /> : null}
    </>
  );
}
