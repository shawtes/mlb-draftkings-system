"use client";

import * as React from "react";
import * as CheckboxPrimitive from "@radix-ui/react-checkbox@1.1.4";
import { CheckIcon } from "lucide-react@0.487.0";

import { cn } from "./utils";

function Checkbox({
  className,
  ...props
}: React.ComponentProps<typeof CheckboxPrimitive.Root>) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      className={cn(
        "peer size-5 shrink-0 rounded-md border border-slate-600 bg-slate-900/80 shadow-xs transition-shadow outline-none focus-visible:ring-[3px] focus-visible:ring-cyan-400/40 focus-visible:border-cyan-300 data-[state=checked]:bg-slate-950 data-[state=checked]:border-cyan-300 data-[state=checked]:shadow-[0_0_10px_rgba(8,145,178,0.45)] disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator
        data-slot="checkbox-indicator"
        className="flex items-center justify-center text-current transition-none"
      >
        <CheckIcon className="size-4 text-black" strokeWidth={3} />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  );
}

export { Checkbox };
