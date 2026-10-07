import { surfaceForHost } from "./security.ts";

export type Surface = "marketing" | "console" | "docs";

export function surfaceFromPath(pathname: string): Surface {
  if (pathname === "/console" || pathname.startsWith("/console/")) return "console";
  if (pathname === "/docs" || pathname.startsWith("/docs/")) return "docs";
  return "marketing";
}

export function pathForSurface(surface: Surface): string {
  return surface === "marketing" ? "/" : `/${surface}`;
}

export function surfaceFromLocation(pathname: string, hostname: string): Surface {
  const hostSurface = surfaceForHost(hostname);
  if (hostSurface === "console" || hostSurface === "docs") return hostSurface;
  return surfaceFromPath(pathname);
}

export function nextHistory(currentPath: string, target: string): { surface: Surface; path: string; push: boolean } {
  const surface = surfaceFromPath(target);
  const path = pathForSurface(surface);
  return { surface, path, push: currentPath !== path };
}
