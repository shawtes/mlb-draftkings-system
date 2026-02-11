"use client";

import * as React from "react";
import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import { CheckIcon } from "lucide-react";

import { cn } from "./utils";

function Checkbox({
  className,
  ...props
}: React.ComponentProps<typeof CheckboxPrimitive.Root>) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      className={cn(
        "peer size-5 shrink-0 rounded-md border-2 border-white/90 bg-white/10 shadow-sm transition-all duration-200 ease-in-out outline-none",
        "hover:border-cyan-400 hover:bg-white/20 hover:shadow-md hover:scale-105",
        "focus-visible:ring-2 focus-visible:ring-cyan-400/50 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 focus-visible:border-cyan-400",
        "data-[state=checked]:bg-gradient-to-br data-[state=checked]:from-cyan-500 data-[state=checked]:to-cyan-600 data-[state=checked]:border-cyan-400 data-[state=checked]:shadow-[0_0_12px_rgba(6,182,212,0.5)] data-[state=checked]:scale-100",
        "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100 disabled:hover:shadow-sm",
        "active:scale-95",
        className,
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator
        data-slot="checkbox-indicator"
        className="flex items-center justify-center text-current transition-all duration-200 ease-in-out animate-in fade-in-0 zoom-in-50"
      >
        <CheckIcon className="size-3.5 text-white font-bold" strokeWidth={3.5} />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  );
}

export { Checkbox };
