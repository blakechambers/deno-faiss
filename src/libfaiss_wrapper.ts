const libExtByOS = {
  darwin: "dylib",
  linux: "so",
  windows: "dll",
  android: "so",
  freebsd: "so",
  openbsd: "so",
  netbsd: "so",
  illumos: "so",
  aix: "dylib",
  solaris: "so",
} as const;
const libExt: string = libExtByOS[Deno.build.os] || "so"; // default to .so for unknown OS

const libPath = new URL(
  `../native/${Deno.build.os}-${Deno.build.arch}/libfaiss_wrapper.${libExt}`,
  import.meta.url,
).pathname;

export const dylibPath = libPath;
