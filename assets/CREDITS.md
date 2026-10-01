# Third-party assets

## Block textures — Isabella II

`content/base/textures/` (except `bedrock.png`, see below) and
`content/base/sky/` come from
[Isabella II](https://github.com/minetest-texture-packs/Isabella-II) (commit
`8afe93927c69`), a texture pack by **Bonemouse**, licensed under the
[Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/).
Original thread: http://www.minecraftforum.net/topic/242175-Isabella/

The pack's `About Isabella.txt` is included unmodified next to the textures.

Changes made for Ruda:

- `copper_ore.png` and `iron_ore.png` are the pack's mineral overlays drawn
  over its stone texture;
- `grass_side.png` is the pack's grass side overlay drawn over its dirt texture;
- `bedrock.png` is the pack's cobblestone texture, darkened and with more contrast;
- `lamp.png` is the pack's `default_meselamp.png`, and `sky/sun.png` and
  `sky/moon.png` are its `misc/sun.png` and `misc/moon.png`, unchanged;
- files are renamed to Ruda's block names.

## Font — Pixelify Sans

`assets/fonts/PixelifySans.ttf` is the Regular weight of
[Pixelify Sans](https://github.com/eifetx/Pixelify-Sans) by **The Pixelify Sans
Project Authors**, licensed under the SIL Open Font License 1.1
(`assets/fonts/OFL.txt`). It is taken from
[pull request #5](https://github.com/eifetx/Pixelify-Sans/pull/5) (commit
`296814deb225`), which adds the Cyrillic capitals О and П missing from the
released font and fixes К.
