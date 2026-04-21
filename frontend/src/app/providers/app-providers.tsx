import type { ReactNode } from "react";
import { MissionStoreProvider } from "@/store/mission-store";

interface AppProvidersProps {
  children: ReactNode;
}

export function AppProviders({ children }: AppProvidersProps): JSX.Element {
  return <MissionStoreProvider>{children}</MissionStoreProvider>;
}
