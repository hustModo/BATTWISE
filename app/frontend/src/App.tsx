import { useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Landing from "@/pages/Landing";
import Workbench from "@/pages/Workbench";
import useSession from "@/store/sessionStore";

export default function App() {
  const bootstrap = useSession((s) => s.bootstrap);
  useEffect(() => {
    bootstrap();
  }, [bootstrap]);

  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/workbench" element={<Workbench />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
