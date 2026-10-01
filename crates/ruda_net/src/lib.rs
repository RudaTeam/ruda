//! Connections between the client and the server.
//!
//! For now there is only the in-memory connection that single-player uses.
//! It still carries serialized messages, so everything that works in
//! single-player goes through the same code as multiplayer will.

use std::fmt;
use std::marker::PhantomData;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::time::Duration;

use ruda_protocol::{ClientMessage, DecodeError, ServerMessage, decode, encode};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// One end of a connection: sends `Out` and receives `In`.
pub struct Connection<Out, In> {
    tx: Sender<Vec<u8>>,
    rx: Receiver<Vec<u8>>,
    messages: PhantomData<fn(Out) -> In>,
}

/// The client's end.
pub type ClientConnection = Connection<ClientMessage, ServerMessage>;
/// The server's end of a connection to one client.
pub type ServerConnection = Connection<ServerMessage, ClientMessage>;

/// Both ends of an in-memory connection.
pub fn local_pair() -> (ClientConnection, ServerConnection) {
    let (to_server, from_client) = mpsc::channel();
    let (to_client, from_server) = mpsc::channel();
    (
        Connection {
            tx: to_server,
            rx: from_server,
            messages: PhantomData,
        },
        Connection {
            tx: to_client,
            rx: from_client,
            messages: PhantomData,
        },
    )
}

impl<Out: Serialize, In: DeserializeOwned> Connection<Out, In> {
    pub fn send(&self, message: &Out) -> Result<(), Disconnected> {
        self.tx.send(encode(message)).map_err(|_| Disconnected)
    }

    /// The next message if one has arrived, without waiting.
    pub fn try_recv(&self) -> Result<Option<In>, RecvError> {
        match self.rx.try_recv() {
            Ok(bytes) => Ok(Some(decode(&bytes)?)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(RecvError::Disconnected),
        }
    }

    /// The next message, waiting up to `timeout` for it.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<In>, RecvError> {
        match self.rx.recv_timeout(timeout) {
            Ok(bytes) => Ok(Some(decode(&bytes)?)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(RecvError::Disconnected),
        }
    }
}

impl<Out, In> fmt::Debug for Connection<Out, In> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Connection(local)")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the other end has disconnected")]
pub struct Disconnected;

#[derive(Debug, thiserror::Error)]
pub enum RecvError {
    #[error("the other end has disconnected")]
    Disconnected,
    #[error(transparent)]
    Malformed(#[from] DecodeError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivers_messages_both_ways_until_one_end_drops() {
        let (client, server) = local_pair();
        let hello = ClientMessage::Hello {
            protocol: 1,
            name: "a".into(),
        };
        client.send(&hello).unwrap();
        assert_eq!(server.try_recv().unwrap(), Some(hello));
        assert_eq!(server.try_recv().unwrap(), None);

        let reply = ServerMessage::ActionDone { seq: 9 };
        server.send(&reply).unwrap();
        assert_eq!(
            client.recv_timeout(Duration::from_secs(1)).unwrap(),
            Some(reply)
        );

        drop(server);
        assert!(matches!(client.try_recv(), Err(RecvError::Disconnected)));
        assert_eq!(
            client.send(&ClientMessage::Position(Default::default())),
            Err(Disconnected)
        );
    }
}
