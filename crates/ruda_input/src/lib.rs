//! Input as game actions rather than keys and buttons, so gameplay code never
//! cares which device produced it and new devices only need bindings.

use std::collections::{HashMap, HashSet};

use glam::Vec2;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Something the player wants to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    /// Also flies up; pressed twice quickly, starts or stops flying.
    Jump,
    /// Also flies down.
    Sneak,
    Sprint,
    Break,
    Place,
    /// Select hotbar slot 0–9.
    Hotbar(u8),
    /// Open the pause menu.
    Pause,
}

/// A physical button that can be bound to an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Button {
    Key(KeyCode),
    Mouse(MouseButton),
}

#[derive(Clone, Debug)]
pub struct Bindings(HashMap<Button, Action>);

impl Default for Bindings {
    fn default() -> Self {
        use Action::*;
        use KeyCode::*;
        let mut bindings: HashMap<Button, Action> = [
            (KeyW, MoveForward),
            (KeyS, MoveBack),
            (KeyA, MoveLeft),
            (KeyD, MoveRight),
            (Space, Jump),
            (ShiftLeft, Sneak),
            (ControlLeft, Sprint),
            (Escape, Action::Pause),
        ]
        .into_iter()
        .map(|(key, action)| (Button::Key(key), action))
        .collect();
        let digits = [
            Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Digit0,
        ];
        for (slot, key) in digits.into_iter().enumerate() {
            bindings.insert(Button::Key(key), Hotbar(slot as u8));
        }
        bindings.insert(Button::Mouse(MouseButton::Left), Break);
        bindings.insert(Button::Mouse(MouseButton::Right), Place);
        Self(bindings)
    }
}

/// The state of every action, fed by window and device events.
#[derive(Debug, Default)]
pub struct Input {
    bindings: Bindings,
    held: HashSet<Action>,
    pressed: Vec<Action>,
    look: Vec2,
    /// Mouse wheel turns, in lines; positive is away from the player.
    scroll: f32,
}

impl Input {
    pub fn new(bindings: Bindings) -> Self {
        Self {
            bindings,
            ..Default::default()
        }
    }

    pub fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } if !event.repeat => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    self.button(Button::Key(key), event.state == ElementState::Pressed);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.button(Button::Mouse(*button), *state == ElementState::Pressed);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, lines) => *lines,
                    // Touchpads scroll in pixels; call a line 40 of them.
                    MouseScrollDelta::PixelDelta(pixels) => pixels.y as f32 / 40.0,
                };
            }
            // Keys released while the window was in the background never
            // report it.
            WindowEvent::Focused(false) => self.held.clear(),
            _ => {}
        }
    }

    pub fn device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (x, y) } = event {
            self.look += Vec2::new(*x as f32, *y as f32);
        }
    }

    /// Presses or releases a button.
    pub fn button(&mut self, button: Button, down: bool) {
        let Some(&action) = self.bindings.0.get(&button) else {
            return;
        };
        if down {
            if self.held.insert(action) {
                self.pressed.push(action);
            }
        } else {
            self.held.remove(&action);
        }
    }

    /// Forgets held buttons and pending presses and movement, for when a
    /// menu takes over the input.
    pub fn clear(&mut self) {
        self.held.clear();
        self.pressed.clear();
        self.look = Vec2::ZERO;
        self.scroll = 0.0;
    }

    pub fn is_held(&self, action: Action) -> bool {
        self.held.contains(&action)
    }

    /// Actions started since the last call, in order.
    pub fn take_pressed(&mut self) -> Vec<Action> {
        std::mem::take(&mut self.pressed)
    }

    /// Mouse movement since the last call, in device units.
    pub fn take_look(&mut self) -> Vec2 {
        std::mem::take(&mut self.look)
    }

    /// Whole mouse wheel steps since the last call, positive away from the
    /// player; the rest of a step carries over.
    pub fn take_scroll(&mut self) -> i32 {
        let steps = self.scroll.trunc();
        self.scroll -= steps;
        steps as i32
    }

    /// Where the player wants to walk: x to the right, y forward, each from
    /// −1 to 1.
    pub fn walk(&self) -> Vec2 {
        let axis = |positive, negative| {
            f32::from(u8::from(self.is_held(positive)))
                - f32::from(u8::from(self.is_held(negative)))
        };
        Vec2::new(
            axis(Action::MoveRight, Action::MoveLeft),
            axis(Action::MoveForward, Action::MoveBack),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_held_and_pressed_actions() {
        let mut input = Input::default();
        input.button(Button::Key(KeyCode::KeyW), true);
        input.button(Button::Key(KeyCode::KeyW), true);
        input.button(Button::Key(KeyCode::KeyD), true);
        assert_eq!(input.walk(), Vec2::new(1.0, 1.0));
        assert_eq!(
            input.take_pressed(),
            vec![Action::MoveForward, Action::MoveRight]
        );
        assert!(input.take_pressed().is_empty());

        input.button(Button::Key(KeyCode::KeyW), false);
        assert_eq!(input.walk(), Vec2::new(1.0, 0.0));
        input.button(Button::Key(KeyCode::Digit3), true);
        assert_eq!(input.take_pressed(), vec![Action::Hotbar(2)]);
    }
}
