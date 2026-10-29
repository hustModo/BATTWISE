import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        battery: {
          bg: "#F6F8FB",
          ink: "#0F2A50",
          lithium: "#C9CFD8",        // silver/grey accents
          electrolyte: "#0EA5A4",     // primary teal
          charge: "#22C55E",          // success / charge green
          amber: "#F59E0B",           // warning
          danger: "#EF4444",          // error
          carbon: "#0B1220",          // dark surface
        },
        agent: {
          router: "#64748B",     // blue-grey
          designer: "#8B5CF6",   // violet
          compute: "#0EA5A4",    // teal
          cluster: "#22C55E",    // green
          refine: "#F59E0B",     // amber
          chat: "#475569",       // slate
        },
      },
      boxShadow: {
        card: "0 4px 24px rgba(15, 42, 80, 0.06)",
        cardHover: "0 10px 32px rgba(15, 42, 80, 0.12)",
      },
      borderRadius: { card: "14px" },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      keyframes: {
        pulse_ring: {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(14, 165, 164, 0.6)" },
          "50%": { boxShadow: "0 0 0 10px rgba(14, 165, 164, 0)" },
        },
      },
      animation: {
        pulse_ring: "pulse_ring 1.8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
