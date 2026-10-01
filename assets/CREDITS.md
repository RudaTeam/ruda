# Third-party assets

## Block textures — Faithful 32x and Faithful PBR 32x

`content/base/textures/`, except `torch.png`, come from
[Faithful 32x](https://faithfulpack.net) by **the Faithful team**
([license](https://faithfulpack.net/license)), with the material maps
(`*_n.png` and `*_s.png`, laid out as in labPBR 1.3) from
[Faithful PBR 32x](https://www.curseforge.com/minecraft/texture-packs/faithful-pbr-32x)
version 1.11 by **PapaChefCool**. They are used in Ruda with permission.

Changes made for Ruda:

- `stone`, `dirt`, `sand`, `grass_top`, `grass_side`, `copper_ore` and
  `iron_ore` put the pack's four variants of each texture (its OptiFine
  "repeat" tiles) together into one 64×64 texture that covers two blocks by
  two, maps alike;
- `cobblestone`, `gravel`, `planks` (the pack's oak planks), `bedrock` and
  `lamp` (its glowstone) are repeated two by two into 64×64;
- `grass_top` and the grass edge of `grass_side` are tinted plains green
  (`#91BD59`); the edge is drawn over the side's earth, and where it covers
  it, its maps replace the earth's;
- files are renamed to Ruda's block names.

## Torch, sun and moon — Isabella II

`content/base/textures/torch.png` and `content/base/sky/` come from
[Isabella II](https://github.com/minetest-texture-packs/Isabella-II) (commit
`8afe93927c69`), a texture pack by **Bonemouse**, licensed under the
[Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/).
Original thread: http://www.minecraftforum.net/topic/242175-Isabella/

The pack's `About Isabella.txt` is included unmodified next to the textures.
`torch.png` is the pack's `default_torch_on_floor.png`, and `sky/sun.png` and
`sky/moon.png` are its `misc/sun.png` and `misc/moon.png`, all unchanged.

## Font — Pixelify Sans

`assets/fonts/PixelifySans.ttf` is the Regular weight of
[Pixelify Sans](https://github.com/eifetx/Pixelify-Sans) by **The Pixelify Sans
Project Authors**, licensed under the SIL Open Font License 1.1
(`assets/fonts/OFL.txt`). It is taken from
[pull request #5](https://github.com/eifetx/Pixelify-Sans/pull/5) (commit
`296814deb225`), which adds the Cyrillic capitals О and П missing from the
released font and fixes К.
