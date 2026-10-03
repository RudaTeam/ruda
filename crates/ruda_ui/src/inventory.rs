//! The inventory: every block there is. As in Minecraft, a block is taken on
//! the pointer by clicking it and put down by clicking a hotbar cell; the
//! same can be done in one movement by dragging it.

use egui::{
    Align2, CursorIcon, DragAndDrop, Id, LayerId, Order, Rect, ScrollArea, Sense, TextEdit, Ui,
    UiBuilder, Vec2, emath::GuiRounding, pos2, vec2,
};

use crate::hud::{CellState, GAP, HotbarSlot, ICON, SLOT, block_name, paint_cell};
use crate::menu::{Menu, MenuAction, MenuContext, dim};
use crate::style::{TEXT, pixel};
use crate::widgets::{EDGE, PANEL_MARGIN, heading, panel, sunken};
use crate::{I18n, InventoryContext};

/// Cells of the catalog and of the hotbar in a row.
const COLUMNS: usize = 10;
/// Rows of the catalog in view; more are scrolled to.
const ROWS_SHOWN: usize = 4;
/// Between a hollow's edge and the cells in it.
const PADDING: f32 = EDGE + 12.0;
const SEARCH_HEIGHT: f32 = 40.0;
const SECTION_GAP: f32 = 12.0;

fn grid_height() -> f32 {
    ROWS_SHOWN as f32 * SLOT + (ROWS_SHOWN as f32 - 1.0) * GAP
}

/// How large the inventory's panel is.
fn panel_size() -> Vec2 {
    let columns = COLUMNS as f32;
    let cells_width = columns * SLOT + (columns - 1.0) * GAP;
    let height = 48.0
        + SECTION_GAP
        + SEARCH_HEIGHT
        + SECTION_GAP
        + grid_height()
        + 2.0 * PADDING
        + SECTION_GAP
        + SLOT
        + 2.0 * PADDING;
    vec2(
        cells_width + 2.0 * PADDING + 2.0 * PANEL_MARGIN,
        height + 2.0 * PANEL_MARGIN,
    )
}

/// Where the catalog's hollow is, in a panel at `panel`.
fn catalog_hollow(panel: Rect) -> Rect {
    let inner = panel.shrink(PANEL_MARGIN);
    let top = inner.min.y + 48.0 + SECTION_GAP + SEARCH_HEIGHT + SECTION_GAP;
    Rect::from_min_size(
        pos2(inner.min.x, top),
        vec2(inner.width(), grid_height() + 2.0 * PADDING),
    )
}

/// Where a cell of the catalog (`place`th among those shown) or of the hotbar
/// is, in a panel at `panel`.
fn cell_rect(panel: Rect, hotbar: bool, place: usize) -> Rect {
    let hollow = catalog_hollow(panel);
    let (top, row, column) = if hotbar {
        (hollow.max.y + SECTION_GAP, 0, place)
    } else {
        (hollow.min.y, place / COLUMNS, place % COLUMNS)
    };
    Rect::from_min_size(
        pos2(
            hollow.min.x + PADDING + column as f32 * (SLOT + GAP),
            top + PADDING + row as f32 * (SLOT + GAP),
        ),
        vec2(SLOT, SLOT),
    )
}

/// What is being dragged.
#[derive(Clone, Copy, Debug)]
enum Drag {
    Catalog(usize),
    Hotbar(usize),
}

/// The name of a block as people read it.
fn name(i18n: &I18n, slot: &HotbarSlot) -> String {
    block_name(i18n, &slot.block)
}

/// Whether a block's name or its id has `query` in it, in any case.
fn matches(i18n: &I18n, slot: &HotbarSlot, query: &str) -> bool {
    let query = query.trim().to_lowercase();
    // The name without the namespace, so searching "base" finds nothing in
    // particular.
    let path = slot.block.rsplit(':').next().unwrap_or_default();
    query.is_empty()
        || name(i18n, slot).to_lowercase().contains(&query)
        || path.to_lowercase().contains(&query)
}

/// Which block of the catalog the hotbar's `slot` holds.
fn catalog_index(inventory: InventoryContext<'_>, slot: &HotbarSlot) -> Option<usize> {
    inventory
        .catalog
        .iter()
        .position(|block| block.block == slot.block)
}

impl Menu {
    pub(crate) fn inventory_screen(
        &mut self,
        ui: &mut Ui,
        context: MenuContext<'_>,
    ) -> Option<MenuAction> {
        dim(ui);
        let (images, i18n, inventory) = (context.images, context.i18n, context.inventory);
        let search_id = Id::new("inventory search");
        let mut query: String = ui.data(|data| data.get_temp(search_id)).unwrap_or_default();
        let mut action = None;
        let carried = &mut self.carried;

        egui::Area::new("inventory".into())
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                let (rect, _) = ui.allocate_exact_size(panel_size(), Sense::hover());
                let rect = rect.round_to_pixels(ui.pixels_per_point());
                panel(ui.painter(), images, rect);
                let inner = rect.shrink(PANEL_MARGIN);

                let title = Rect::from_min_size(inner.min, vec2(inner.width(), 48.0));
                heading(ui.painter(), title.center(), &i18n.get("inventory-title"));

                // The search field.
                let search = Rect::from_min_size(
                    pos2(inner.min.x, title.max.y + SECTION_GAP),
                    vec2(inner.width(), SEARCH_HEIGHT),
                );
                sunken(ui.painter(), search);
                ui.put(
                    search.shrink2(vec2(EDGE + 8.0, 4.0)),
                    TextEdit::singleline(&mut query)
                        .frame(egui::Frame::NONE)
                        .hint_text(i18n.get("inventory-search"))
                        .font(pixel(24.0))
                        .text_color(TEXT)
                        .desired_width(f32::INFINITY)
                        .vertical_align(egui::Align::Center),
                );

                // The catalog, narrowed to what the search asks for. A click
                // between the cells puts a block in hand back.
                let hollow = catalog_hollow(rect);
                sunken(ui.painter(), hollow);
                let background = ui.interact(hollow, Id::new("catalog hollow"), Sense::click());
                if background.clicked() {
                    *carried = None;
                }
                let shown: Vec<usize> = (0..inventory.catalog.len())
                    .filter(|&index| matches(i18n, &inventory.catalog[index], &query))
                    .collect();
                if shown.is_empty() {
                    ui.painter().text(
                        hollow.center(),
                        Align2::CENTER_CENTER,
                        i18n.get("inventory-empty"),
                        pixel(24.0),
                        TEXT.gamma_multiply(0.5),
                    );
                }
                ui.scope_builder(UiBuilder::new().max_rect(hollow.shrink(PADDING)), |ui| {
                    ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                        catalog_cells(ui, &shown, inventory, carried);
                    });
                });

                // The hotbar.
                let hotbar = Rect::from_min_size(
                    pos2(inner.min.x, hollow.max.y + SECTION_GAP),
                    vec2(inner.width(), SLOT + 2.0 * PADDING),
                );
                sunken(ui.painter(), hotbar);
                for (index, content) in inventory.hotbar.iter().enumerate() {
                    let cell = cell_rect(rect, true, index);
                    let response = ui.interact(
                        cell,
                        Id::new(("hotbar cell", index)),
                        Sense::click_and_drag(),
                    );
                    if content.is_some() && carried.is_none() {
                        response.dnd_set_drag_payload(Drag::Hotbar(index));
                    }
                    let holding = response.dnd_hover_payload::<Drag>().is_some();
                    if let Some(drag) = response.dnd_release_payload::<Drag>() {
                        action = match *drag {
                            Drag::Catalog(block) => Some(MenuAction::SetHotbar {
                                slot: index,
                                block: Some(block),
                            }),
                            Drag::Hotbar(from) if from != index => {
                                Some(MenuAction::SwapHotbar(from, index))
                            }
                            Drag::Hotbar(_) => action,
                        };
                    } else if response.clicked() {
                        let there = content
                            .as_ref()
                            .and_then(|slot| catalog_index(inventory, slot));
                        if let Some(block) = carried.take() {
                            // Put it down; what was there is in hand now.
                            *carried = there;
                            action = Some(MenuAction::SetHotbar {
                                slot: index,
                                block: Some(block),
                            });
                        } else if there.is_some() {
                            // Take what is there, leaving the cell empty.
                            *carried = there;
                            action = Some(MenuAction::SetHotbar {
                                slot: index,
                                block: None,
                            });
                        }
                    }
                    let state = if index == inventory.selected {
                        CellState::Selected
                    } else if response.hovered() || holding {
                        CellState::Hovered
                    } else {
                        CellState::Plain
                    };
                    let icon = content.as_ref().and_then(|slot| slot.icon.as_ref());
                    paint_cell(ui.painter(), cell, icon, Some(index), state);
                }

                // A click outside the panel puts a block in hand away too.
                let outside = ui.input(|input| {
                    input.pointer.any_click()
                        && input
                            .pointer
                            .interact_pos()
                            .is_some_and(|at| !rect.contains(at))
                });
                if outside {
                    *carried = None;
                }
                carried_block(ui, inventory, *carried);
            });

        ui.data_mut(|data| data.insert_temp(search_id, query));
        action
    }
}

/// The cells of the catalog in `shown`, in rows.
fn catalog_cells(
    ui: &mut Ui,
    shown: &[usize],
    inventory: InventoryContext<'_>,
    carried: &mut Option<usize>,
) {
    let rows = shown.len().div_ceil(COLUMNS).max(ROWS_SHOWN);
    let size = vec2(
        COLUMNS as f32 * SLOT + (COLUMNS as f32 - 1.0) * GAP,
        rows as f32 * SLOT + (rows as f32 - 1.0) * GAP,
    );
    let (area, _) = ui.allocate_exact_size(size, Sense::hover());
    for (place, &index) in shown.iter().enumerate() {
        let (row, column) = (place / COLUMNS, place % COLUMNS);
        let cell = Rect::from_min_size(
            area.min + vec2(column as f32 * (SLOT + GAP), row as f32 * (SLOT + GAP)),
            vec2(SLOT, SLOT),
        );
        let slot = &inventory.catalog[index];
        let response = ui.interact(
            cell,
            Id::new(("catalog cell", index)),
            Sense::click_and_drag(),
        );
        if carried.is_none() {
            response.dnd_set_drag_payload(Drag::Catalog(index));
        }
        if response.clicked() {
            // Take it, or if a block is in hand already, put that back.
            *carried = if carried.is_some() { None } else { Some(index) };
        }
        let state = if response.hovered() {
            CellState::Hovered
        } else {
            CellState::Plain
        };
        paint_cell(ui.painter(), cell, slot.icon.as_ref(), None, state);
    }
}

/// The block in hand or being dragged, following the pointer.
fn carried_block(ui: &Ui, inventory: InventoryContext<'_>, carried: Option<usize>) {
    let ctx = ui.ctx();
    let slot = match (carried, DragAndDrop::payload::<Drag>(ctx)) {
        (Some(index), _) => inventory.catalog.get(index),
        (None, Some(drag)) => match *drag {
            Drag::Catalog(index) => inventory.catalog.get(index),
            Drag::Hotbar(index) => inventory.hotbar.get(index).and_then(Option::as_ref),
        },
        (None, None) => None,
    };
    let (Some(icon), Some(pointer)) = (
        slot.and_then(|slot| slot.icon.as_ref()),
        ctx.pointer_interact_pos(),
    ) else {
        return;
    };
    ctx.set_cursor_icon(CursorIcon::Grabbing);
    let painter = ctx.layer_painter(LayerId::new(Order::Tooltip, Id::new("carried block")));
    let uv = Rect::from_min_max(egui::Pos2::ZERO, pos2(1.0, 1.0));
    painter.image(
        icon.id(),
        Rect::from_center_size(pointer, vec2(ICON, ICON)),
        uv,
        egui::Color32::WHITE.gamma_multiply(0.9),
    );
}

#[cfg(test)]
mod tests {
    use egui::{Context, Event, Modifiers, PointerButton, Pos2, RawInput};

    use super::*;
    use crate::widgets::Images;
    use crate::{Language, Screen};

    fn slot(name: &str) -> HotbarSlot {
        HotbarSlot {
            block: format!("base:{name}"),
            icon: None,
        }
    }

    /// An egui context, a menu over it and an inventory of three blocks,
    /// with stone and dirt in the first two cells of a hotbar of three, and
    /// the second in hand.
    struct Table {
        ctx: Context,
        menu: Menu,
        i18n: I18n,
        images: Images,
        catalog: Vec<HotbarSlot>,
        hotbar: Vec<Option<HotbarSlot>>,
        time: f64,
    }

    impl Table {
        fn new() -> Self {
            let ctx = Context::default();
            crate::style::apply(&ctx);
            Self {
                ctx,
                menu: Menu::new(Screen::Inventory),
                i18n: I18n::new(Language::English),
                images: Images::default(),
                catalog: vec![slot("stone"), slot("dirt"), slot("torch")],
                hotbar: vec![Some(slot("stone")), Some(slot("dirt")), None],
                time: 0.0,
            }
        }

        fn screen() -> Rect {
            Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))
        }

        fn panel() -> Rect {
            Rect::from_center_size(Self::screen().center(), panel_size())
        }

        /// A frame with `events`; what the menu asked for, if anything.
        fn frame(&mut self, events: Vec<Event>) -> Option<MenuAction> {
            self.time += 0.05;
            let input = RawInput {
                screen_rect: Some(Self::screen()),
                time: Some(self.time),
                events,
                ..Default::default()
            };
            let context = MenuContext {
                i18n: &self.i18n,
                images: &self.images,
                inventory: InventoryContext {
                    catalog: &self.catalog,
                    hotbar: &self.hotbar,
                    selected: 1,
                },
                max_scale: 100,
                world_visible: false,
                version: "test",
                system_language: Language::English,
            };
            let mut settings = crate::Settings::default();
            let mut action = None;
            let menu = &mut self.menu;
            let mut output = self.ctx.run_ui(input, |ui| {
                action = menu.show(ui, context, &mut settings);
            });
            // Nothing draws these here.
            output.textures_delta.clear();
            action
        }

        fn move_to(&mut self, at: Pos2) -> Option<MenuAction> {
            self.frame(vec![Event::PointerMoved(at)])
        }

        fn button(&mut self, at: Pos2, pressed: bool) -> Option<MenuAction> {
            self.frame(vec![Event::PointerButton {
                pos: at,
                button: PointerButton::Primary,
                pressed,
                modifiers: Modifiers::NONE,
            }])
        }

        /// Clicks and lets go.
        fn click(&mut self, at: Pos2) -> Option<MenuAction> {
            self.move_to(at);
            self.button(at, true);
            self.button(at, false)
        }

        /// Takes what is at `from` and lets go of it at `to`.
        fn drag(&mut self, from: Pos2, to: Pos2) -> Option<MenuAction> {
            self.move_to(from);
            self.button(from, true);
            let middle = from.lerp(to, 0.5);
            self.move_to(middle);
            self.move_to(to);
            self.button(to, false)
        }
    }

    fn catalog_cell(place: usize) -> Pos2 {
        cell_rect(Table::panel(), false, place).center()
    }

    fn hotbar_cell(place: usize) -> Pos2 {
        cell_rect(Table::panel(), true, place).center()
    }

    /// Somewhere on the screen that is not the panel.
    fn outside() -> Pos2 {
        pos2(20.0, 20.0)
    }

    #[test]
    fn clicking_a_block_takes_it_on_the_pointer_and_puts_nothing_anywhere() {
        let mut table = Table::new();
        table.frame(vec![]);
        assert_eq!(table.click(catalog_cell(2)), None);
        assert_eq!(table.menu.carried, Some(2));
    }

    #[test]
    fn a_block_in_hand_goes_in_the_empty_cell_clicked() {
        let mut table = Table::new();
        table.frame(vec![]);
        table.click(catalog_cell(2));
        assert_eq!(
            table.click(hotbar_cell(2)),
            Some(MenuAction::SetHotbar {
                slot: 2,
                block: Some(2)
            })
        );
        assert_eq!(table.menu.carried, None);
    }

    #[test]
    fn a_block_put_on_a_full_cell_leaves_the_old_one_in_hand() {
        let mut table = Table::new();
        table.frame(vec![]);
        table.click(catalog_cell(2));
        assert_eq!(
            table.click(hotbar_cell(1)),
            Some(MenuAction::SetHotbar {
                slot: 1,
                block: Some(2)
            })
        );
        // Dirt was in the cell: it is the second of the catalog.
        assert_eq!(table.menu.carried, Some(1));
    }

    #[test]
    fn clicking_a_full_cell_with_nothing_in_hand_takes_its_block() {
        let mut table = Table::new();
        table.frame(vec![]);
        assert_eq!(
            table.click(hotbar_cell(0)),
            Some(MenuAction::SetHotbar {
                slot: 0,
                block: None
            })
        );
        assert_eq!(table.menu.carried, Some(0));
    }

    #[test]
    fn clicking_an_empty_cell_with_nothing_in_hand_does_nothing() {
        let mut table = Table::new();
        table.frame(vec![]);
        assert_eq!(table.click(hotbar_cell(2)), None);
        assert_eq!(table.menu.carried, None);
    }

    #[test]
    fn a_block_in_hand_is_put_away_by_a_click_elsewhere() {
        for away in [outside(), catalog_cell(8), catalog_cell(1)] {
            let mut table = Table::new();
            table.frame(vec![]);
            table.click(catalog_cell(0));
            assert_eq!(table.menu.carried, Some(0));
            assert_eq!(table.click(away), None, "{away:?}");
            assert_eq!(table.menu.carried, None, "{away:?}");
        }
    }

    #[test]
    fn dragging_a_block_to_a_cell_puts_it_there() {
        let mut table = Table::new();
        table.frame(vec![]);
        assert_eq!(
            table.drag(catalog_cell(2), hotbar_cell(0)),
            Some(MenuAction::SetHotbar {
                slot: 0,
                block: Some(2)
            })
        );
        assert_eq!(table.menu.carried, None);
    }

    #[test]
    fn dragging_one_cell_onto_another_swaps_them() {
        let mut table = Table::new();
        table.frame(vec![]);
        assert_eq!(
            table.drag(hotbar_cell(0), hotbar_cell(1)),
            Some(MenuAction::SwapHotbar(0, 1))
        );
        // An empty cell has nothing to drag.
        assert_eq!(table.drag(hotbar_cell(2), hotbar_cell(0)), None);
    }

    #[test]
    fn letting_go_of_a_block_anywhere_else_does_nothing() {
        let mut table = Table::new();
        table.frame(vec![]);
        assert_eq!(table.drag(catalog_cell(0), outside()), None);
        assert_eq!(table.menu.carried, None);
    }

    #[test]
    fn the_search_narrows_the_catalog() {
        let i18n = I18n::new(Language::English);
        let catalog = [slot("stone"), slot("cobblestone"), slot("torch")];
        let found = |query: &str| -> Vec<&str> {
            catalog
                .iter()
                .filter(|slot| matches(&i18n, slot, query))
                .map(|slot| slot.block.trim_start_matches("base:"))
                .collect()
        };
        assert_eq!(found(""), ["stone", "cobblestone", "torch"]);
        assert_eq!(found("STONE"), ["stone", "cobblestone"]);
        assert_eq!(found("  torch "), ["torch"]);
        assert!(found("diamond").is_empty());
        // In the language of the player too.
        let russian = I18n::new(Language::Russian);
        assert!(matches(&russian, &catalog[2], "факел"));
    }
}
