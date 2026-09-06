export class ServiceError extends Error {
  constructor(public readonly status: number, public readonly code: string, message: string) {
    super(message);
    this.name = "ServiceError";
  }
}

export function invariant(condition: unknown, status: number, code: string, message: string): asserts condition {
  if (!condition) throw new ServiceError(status, code, message);
}
