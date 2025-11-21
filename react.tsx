import { useEffect, useState } from "react";

export default function TextFromAssets() {
  const [text, setText] = useState<string>("");

  useEffect(() => {
    fetch("/assets/readme.txt") // file at public/assets/readme.txt
      .then((r) => r.text())
      .then(setText)
      .catch((err) => console.error("Failed to load text:", err));
  }, []);

  return <pre>{text}</pre>;
}