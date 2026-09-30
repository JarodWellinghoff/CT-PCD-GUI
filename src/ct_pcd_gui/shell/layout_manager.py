from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget
from ct_pcd_gui.shell.view_panes import SliceViewPane, VIEW_COLORS, ThreeDViewPane


class LayoutManager(QWidget):
    """Owns the panes and rearranges them into named layouts.

    Panes are created once and reparented, so a renderer living inside a pane
    survives layout switches.
    """

    LAYOUTS = (
        "Four-Up",
        "Conventional",
        "Two-by-Two",
        "Red Slice Only",
        "3D Only",
        "Side by Side",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stash = QWidget(self)  # holds panes not currently shown
        self._stash.hide()
        self._root = None

        self.panes = {
            "Red": SliceViewPane("Red", VIEW_COLORS["Red"], "R", "S"),
            "Yellow": SliceViewPane("Yellow", VIEW_COLORS["Yellow"], "Y", "L"),
            "Green": SliceViewPane("Green", VIEW_COLORS["Green"], "G", "A"),
            "3D": ThreeDViewPane(),
        }
        for p in self.panes.values():
            p.setParent(self._stash)

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self.set_layout_name("Four-Up")

    # -- helpers ----------------------------------------------------------
    def _split(self, orientation, widgets, sizes=None):
        s = QSplitter(orientation)
        s.setHandleWidth(4)
        s.setChildrenCollapsible(False)
        for w in widgets:
            s.addWidget(w)
        if sizes:
            s.setSizes(sizes)
        return s

    def _detach_all(self):
        for p in self.panes.values():
            p.setParent(self._stash)
            p.hide()
        if self._root is not None:
            self._root.setParent(None)
            self._root.deleteLater()
            self._root = None

    def _install(self, root, shown):
        self._root = root
        self._outer.addWidget(root)
        for name in shown:
            self.panes[name].show()
        root.show()

    # -- public -----------------------------------------------------------
    def set_layout_name(self, name):
        self._detach_all()
        P = self.panes

        if name == "Four-Up":
            top = self._split(Qt.Orientation.Horizontal, [P["Red"], P["3D"]])
            bottom = self._split(Qt.Orientation.Horizontal, [P["Yellow"], P["Green"]])
            root = self._split(Qt.Orientation.Vertical, [top, bottom], sizes=[400, 400])
            shown = ["Red", "3D", "Yellow", "Green"]

        elif name == "Conventional":
            slices = self._split(
                Qt.Orientation.Horizontal, [P["Red"], P["Yellow"], P["Green"]]
            )
            root = self._split(
                Qt.Orientation.Vertical, [P["3D"], slices], sizes=[500, 260]
            )
            shown = ["3D", "Red", "Yellow", "Green"]

        elif name == "Two-by-Two":
            top = self._split(Qt.Orientation.Horizontal, [P["Red"], P["Yellow"]])
            bottom = self._split(Qt.Orientation.Horizontal, [P["Green"], P["3D"]])
            root = self._split(Qt.Orientation.Vertical, [top, bottom], sizes=[400, 400])
            shown = ["Red", "Yellow", "Green", "3D"]

        elif name == "Side by Side":
            root = self._split(Qt.Orientation.Horizontal, [P["Red"], P["3D"]])
            shown = ["Red", "3D"]

        elif name == "Red Slice Only":
            root = self._split(Qt.Orientation.Horizontal, [P["Red"]])
            shown = ["Red"]

        elif name == "3D Only":
            root = self._split(Qt.Orientation.Horizontal, [P["3D"]])
            shown = ["3D"]
        else:
            raise ValueError(f"Unknown layout: {name}")
        self._install(root, shown)
