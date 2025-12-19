import { api, HydrateClient } from "~/trpc/server";

export default async function Home() {
  return (
    <HydrateClient>
      <main>
        <div>Flex AI</div>
      </main>
    </HydrateClient>
  );
}
