//! Game rules shared by the client and the server.
//!
//! The server runs them to decide what happens. The client runs the same
//! code to show its player the result of their input right away instead of
//! waiting for the server; given the same input and the same world, both
//! come to the same result, and where they don't, the server's word wins.

mod movement;

pub use movement::{
    EYE_HEIGHT, FLY_CEILING, PLAYER_HEIGHT, PLAYER_WIDTH, PlayerInput, PlayerState, solid_in,
};
