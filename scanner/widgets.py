"""
Reusable custom Tk widgets for the HMAxEMA Scanner GUI.
Font-independent vector widgets drawn with tk.Canvas.
"""

import tkinter as tk


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class GradientCanvas(tk.Canvas):
    """Tk canvas that paints a smooth horizontal gradient behind its children."""

    def set_gradient(self, hex_colors, horizontal: bool = True):
        self._colors = [_hex_to_rgb(c) for c in hex_colors]
        self._horizontal = horizontal
        self._paint()

    def _paint(self):
        if not hasattr(self, "_colors") or len(self._colors) < 2:
            return
        self.delete("grad")
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        steps = max((w if self._horizontal else h) // 2, 1)
        n = len(self._colors) - 1
        for i in range(steps):
            t = i / max(steps - 1, 1) * n
            seg = min(int(t), n - 1)
            f = t - seg
            c1, c2 = self._colors[seg], self._colors[seg + 1]
            rgb = tuple(int(c1[j] + (c2[j] - c1[j]) * f) for j in range(3))
            col = "#%02x%02x%02x" % rgb
            if self._horizontal:
                self.create_line(i * 2, 0, i * 2, h, width=3, fill=col, tags="grad")
            else:
                self.create_line(0, i * 2, w, i * 2, width=3, fill=col, tags="grad")
        self.tag_lower("grad")


class AvatarRing(tk.Canvas):
    """Decorative multi-color ring around a circle label — like the mock's avatar."""

    def __init__(self, master, size=84, letter="H", bg="#0c1e13",
                 ring_colors=("#00ddcc", "#00ff88", "#aaff00"), **kw):
        super().__init__(master, width=size, height=size, bg=bg,
                         highlightthickness=0, bd=0, **kw)
        self._size = size
        self._letter = letter
        self._ring = ring_colors
        self.bind("<Configure>", lambda e: self._draw())

    def _draw(self):
        self.delete("all")
        s = min(self.winfo_width(), self.winfo_height()) or self._size
        cx = cy = s / 2
        r_out = s / 2 - 3
        for k, col in enumerate(self._ring):
            start = 15 + k * 110
            extent = 200 + k * 35
            self.create_arc(cx - r_out, cy - r_out, cx + r_out, cy + r_out,
                            start=start, extent=extent, style="arc",
                            outline=col, width=2)
        r_in = s / 2 - 12
        self.create_oval(cx - r_in, cy - r_in, cx + r_in, cy + r_in,
                         fill="#12331f", outline="#1a4a2a", width=1)
        self.create_text(cx, cy, text=self._letter,
                         font=("Segoe UI", int(s / 3.4), "bold"), fill="#8dffc4")


class ToolTip:
    """Minimal tooltip for icon buttons."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)
        lbl = tk.Label(self.tip, text=self.text, bg="#0f2a1a", fg="#c8d8c0",
                       font=("Segoe UI", 9), padx=8, pady=4)
        lbl.pack()
        self.tip.wm_geometry(f"+{x}+{y}")

    def _hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None
