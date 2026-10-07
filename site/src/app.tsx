import { useEffect, useState } from "react";
import { Console } from "./console.tsx";
import { Docs } from "./docs.tsx";
import { Marketing } from "./marketing.tsx";
import { nextHistory, surfaceFromLocation, type Surface } from "./navigate.ts";

function currentSurface(): Surface {
  return surfaceFromLocation(location.pathname, location.hostname);
}

export function App() {
  const [surface, setSurface] = useState<Surface>(currentSurface);
  const [theme, setTheme] = useState("light");
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(() => {
    function onPop() {
      setSurface(currentSurface());
    }
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  function navigate(path: string) {
    const step = nextHistory(location.pathname, path);
    if (step.push) history.pushState({ surface: step.surface }, "", step.path);
    setSurface(step.surface);
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
