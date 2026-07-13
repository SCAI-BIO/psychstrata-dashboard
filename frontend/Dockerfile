FROM node:22-alpine AS base

ENV PNPM_HOME="/pnpm" \
    PATH="/pnpm:$PATH"

WORKDIR /app

RUN corepack enable

FROM base AS deps

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM deps AS dev

COPY . .

EXPOSE 5173

CMD ["pnpm", "dev", "--host", "0.0.0.0", "--port", "5173"]

FROM deps AS build

ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_APP_PASSWORD=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_APP_PASSWORD=${VITE_APP_PASSWORD}

COPY . .
RUN pnpm build

FROM nginx:1.27-alpine AS production

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
